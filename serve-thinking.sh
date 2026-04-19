#!/usr/bin/env bash
set -euo pipefail

MODEL="$HOME/.lmstudio/models/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
PORT=8081

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Server already running on port $PORT"
  exit 0
fi

exec rapid-mlx serve "$MODEL" \
  --port "$PORT" \
  --served-model-name qwen3.6-35b-a3b \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --kv-bits 4 \
  --kv-cache-quantization
