// Copyright 2026 Teradata
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package embedding

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// NewOpenAIEmbedder — default filling
// ---------------------------------------------------------------------------

func TestNewOpenAIEmbedder_Defaults(t *testing.T) {
	e := NewOpenAIEmbedder(OpenAIConfig{APIKey: "test-key"})
	assert.Equal(t, "text-embedding-3-small", e.config.Model)
	assert.Equal(t, 1536, e.config.Dimensions)
	assert.Equal(t, "https://api.openai.com/v1", e.config.BaseURL)
	assert.Equal(t, 30*time.Second, e.config.Timeout)
	assert.Equal(t, 30*time.Second, e.client.Timeout)
}

func TestNewOpenAIEmbedder_ExplicitValues(t *testing.T) {
	cfg := OpenAIConfig{
		APIKey:     "key",
		Model:      "text-embedding-ada-002",
		Dimensions: 512,
		BaseURL:    "https://custom.example.com/v1",
		Timeout:    10 * time.Second,
	}
	e := NewOpenAIEmbedder(cfg)
	assert.Equal(t, "text-embedding-ada-002", e.config.Model)
	assert.Equal(t, 512, e.config.Dimensions)
	assert.Equal(t, "https://custom.example.com/v1", e.config.BaseURL)
	assert.Equal(t, 10*time.Second, e.config.Timeout)
}

// ---------------------------------------------------------------------------
// EmbedBatch — helpers
// ---------------------------------------------------------------------------

func makeServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func embedderWithServer(t *testing.T, srv *httptest.Server) *OpenAIEmbedder {
	t.Helper()
	return NewOpenAIEmbedder(OpenAIConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
}

type testEmbeddingResponse struct {
	Data  []map[string]interface{} `json:"data"`
	Error *map[string]string       `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// EmbedBatch — success cases
// ---------------------------------------------------------------------------

func TestEmbedBatch_Success(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "Bearer test-key", r.Header.Get("Authorization"))

		resp := testEmbeddingResponse{
			Data: []map[string]interface{}{
				{"index": 0, "embedding": []float32{0.1, 0.2, 0.3}},
				{"index": 1, "embedding": []float32{0.4, 0.5, 0.6}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	e := embedderWithServer(t, srv)
	results, err := e.EmbedBatch(context.Background(), []string{"hello", "world"})
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.InDeltaSlice(t, []float32{0.1, 0.2, 0.3}, results[0], 1e-5)
	assert.InDeltaSlice(t, []float32{0.4, 0.5, 0.6}, results[1], 1e-5)
}

func TestEmbedBatch_EmptyInput_ReturnsNil(t *testing.T) {
	// No server needed — empty input short-circuits before any HTTP call.
	e := NewOpenAIEmbedder(OpenAIConfig{APIKey: "key"})
	results, err := e.EmbedBatch(context.Background(), nil)
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestEmbedBatch_PreservesOrder(t *testing.T) {
	// Server returns embeddings in reverse index order to confirm sorting.
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		resp := testEmbeddingResponse{
			Data: []map[string]interface{}{
				{"index": 1, "embedding": []float32{1.0}},
				{"index": 0, "embedding": []float32{0.0}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	e := embedderWithServer(t, srv)
	results, err := e.EmbedBatch(context.Background(), []string{"a", "b"})
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.InDelta(t, float32(0.0), results[0][0], 1e-5, "index 0 should map to first input")
	assert.InDelta(t, float32(1.0), results[1][0], 1e-5, "index 1 should map to second input")
}

// ---------------------------------------------------------------------------
// EmbedBatch — error cases
// ---------------------------------------------------------------------------

func TestEmbedBatch_HTTPError(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal server error", http.StatusInternalServerError)
	})
	e := embedderWithServer(t, srv)
	_, err := e.EmbedBatch(context.Background(), []string{"text"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "500")
}

func TestEmbedBatch_APIErrorInResponse(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"error": map[string]string{
				"message": "invalid api key",
				"type":    "authentication_error",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	e := embedderWithServer(t, srv)
	_, err := e.EmbedBatch(context.Background(), []string{"text"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid api key")
}

func TestEmbedBatch_ContextCancelled(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Hang until the client gives up.
		<-r.Context().Done()
	})
	e := embedderWithServer(t, srv)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately
	_, err := e.EmbedBatch(ctx, []string{"text"})
	require.Error(t, err)
}

// ---------------------------------------------------------------------------
// Embed — delegates to EmbedBatch
// ---------------------------------------------------------------------------

func TestEmbed_Success(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		resp := testEmbeddingResponse{
			Data: []map[string]interface{}{
				{"index": 0, "embedding": []float32{0.7, 0.8, 0.9}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	e := embedderWithServer(t, srv)
	vec, err := e.Embed(context.Background(), "single text")
	require.NoError(t, err)
	assert.InDeltaSlice(t, []float32{0.7, 0.8, 0.9}, vec, 1e-5)
}

func TestEmbed_Error(t *testing.T) {
	srv := makeServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	})
	e := embedderWithServer(t, srv)
	_, err := e.Embed(context.Background(), "text")
	require.Error(t, err)
}
