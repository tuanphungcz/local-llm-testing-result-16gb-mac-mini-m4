# Local LLM Testing on Apple Silicon

Real-world benchmarks for running local LLMs on Apple Silicon Macs. Two machines, multiple runtimes, multiple models, measured — not estimated.

**All numbers measured on real hardware.** macOS 26.2, April 2026.

---

## TL;DR

| Machine | Best daily driver | Best speed | Best long context |
|---------|------------------|------------|-------------------|
| **MacBook Pro M3 Max, 36 GB** | llama.cpp IQ4_XS + LPM (33.8 tok/s, 65.8°C) | Rapid-MLX 4-bit, no LPM (62.3 tok/s) | llama.cpp IQ3_XXS + LPM (16–44 tok/s, 128K ctx) |
| **Mac mini M4, 16 GB** | Qwen 3.5 4B MLX 4-bit (38 tok/s, 2.9 GB) | Gemma 4 E2B MLX 4-bit (64.5 tok/s) | Qwen 3.5 9B tuned → 56K ctx |

**Key insight for 36 GB:** Low Power Mode (`sudo pmset -a lowpowermode 1`) drops GPU power 57–76% for only a 22–33% speed loss. LLM inference is memory-bandwidth bound — throttling compute barely hurts.

**Key insight for 16 GB:** MLX beats llama.cpp by ~25% on models under 14B. Metal OOM is the real context limit — published context sizes are meaningless if your GPU can't hold the KV cache. Pick the smallest model that does the job.

---

## Hardware

| | MacBook Pro | Mac mini |
|---|---|---|
| **Chip** | M3 Max | M4 |
| **RAM** | 36 GB unified | 16 GB unified |
| **Memory bandwidth** | ~410 GB/s | ~120 GB/s |
| **GPU cores** | 30 | 10 |
| **Use case** | Laptop — thermals matter | Desktop — thermals not a concern |

Memory bandwidth is the hard ceiling for LLM inference speed. M4 Max (410 GB/s) ≈ M3 Max ≈ 2× M4 Pro (273 GB/s) ≈ 3.4× M4 base (120 GB/s).

---

## Benchmarks

### Generation speed (tok/s) — short prompt, ~200–600 token generation

#### M3 Max 36 GB

| Runtime | Model | Quant | LPM | Gen tok/s | GPU max °C | GPU peak W |
|---------|-------|-------|-----|-----------|------------|------------|
| llama.cpp | Qwen 3.6 35B-A3B | IQ4_XS GGUF | OFF | 43.5 | 95 | 42 |
| llama.cpp | Qwen 3.6 35B-A3B | IQ4_XS GGUF | ON | 33.8 | 65.8 | 10.6 |
| llama.cpp | Qwen 3.6 35B-A3B | IQ3_XXS GGUF | ON | 36.7 | 56.8 | 12.5 |
| Rapid-MLX | Qwen 3.6 35B-A3B | MLX 3-bit | OFF | 63.0 | 82.0 | 22.5 |
| Rapid-MLX | Qwen 3.6 35B-A3B | MLX 3-bit | ON | 43.8 | 64.9 | 9.5 |
| Rapid-MLX | Qwen 3.6 35B-A3B | MLX 4-bit | OFF | 62.3 | 76.0 | 21.5 |
| Rapid-MLX | Qwen 3.6 35B-A3B | MLX 4-bit | ON | 41.8 | 68.4 | 9.3 |
| oMLX | Qwen 3.6 35B-A3B | MLX 3-bit | OFF | 42.4 | 65.4 | 18.1 |
| oMLX | Qwen 3.6 35B-A3B | MLX 3-bit | ON | 30.6 | 61.6 | 9.7 |

#### Mac mini M4 16 GB

| Runtime | Model | Quant | Gen tok/s | Prompt tok/s | RAM | Note |
|---------|-------|------|-----------|-------------|-----|------|
| MLX | Qwen 3.5 4B | 4-bit | **38.1** | 363 | 2.9 GB | Best balance |
| MLX | Qwen 3.5 9B | 4-bit | 21.1 | 224 | 5.6 GB | Solid all-rounder |
| llama.cpp | Qwen 3.5 9B | Q4_K_M | 17.1 | 209 | 5.3 GB | Stable Metal offload |
| llama.cpp | Qwen 3.5 35B MoE | UD-IQ3_XXS | 16.5 | 62 | 12 GB | CPU + mmap, swaps |
| MLX | Gemma 4 E2B | 4-bit | **64.5** | 2249 | 3.4 GB | Fastest small |
| MLX | Gemma 4 E4B | 8-bit | 19.2 | 601 | 8.4 GB | Fast prompt eval |
| MLX | Gemma 4 26B | UD-3bit | 31.2 | 259 | 12 GB | OOM >200 tok prompt |
| ollama | qwen3:8b | — | 19.4 | — | — | Earlier measure |

### LPM cost (M3 Max)

| Runtime | Speed loss | Temp drop | Power drop |
|---------|------------|-----------|------------|
| llama.cpp IQ4_XS | −22% (43.5→33.8) | −29°C (95→65.8) | −76% (42→10 W) |
| Rapid-MLX 4-bit | −33% (62.3→41.8) | −8°C (76→68.4) | −57% (21.5→9.3 W) |
| Rapid-MLX 3-bit | −31% (63→43.8) | −17°C (82→64.9) | −58% (22.5→9.5 W) |

### Quantization tier (M3 Max, Rapid-MLX)

| Quant | Size | LPM | Gen tok/s | GPU max °C | Quality (KLD) |
|-------|------|-----|-----------|------------|---------------|
| 3-bit | 17 GB | OFF | 63.0 | 82.0 | ~0.08 |
| 4-bit | 20 GB | OFF | 62.3 | 76.0 | ~0.03 |
| 3-bit | 17 GB | ON | 43.8 | 64.9 | ~0.08 |
| 4-bit | 20 GB | ON | 41.8 | 68.4 | ~0.03 |

4-bit is the same speed as 3-bit and cooler. No reason to run 3-bit on 36 GB.

### Context scaling (M3 Max)

#### llama.cpp IQ3_XXS, Q8 KV, ctx 128K, LPM ON — best for long context

| Input tokens | Gen tok/s | Prompt tok/s | Elapsed |
|-------------|-----------|-------------|---------|
| 1,684 | 44.0 | 497 | 3.7 s |
| 8,334 | 41.0 | 523 | 14 s |
| 16,684 | 38.3 | 456 | 20 s |
| 33,334 | 28.9 | 336 | 52 s |
| 50,034 | 24.9 | 246 | 71 s |
| 83,334 | 19.0 | 167 | 203 s |
| 106,684 | 16.0 | 112 | 215 s |
| 133k | 400 Bad Request (over ctx) | — | — |

#### Context ceiling by runtime (M3 Max)

| Runtime | Quant | KV | Max working context | Gen tok/s @ 50K | Gen tok/s @ 100K |
|---------|-------|-----|---------------------|-----------------|-------------------|
| Rapid-MLX | MLX 4-bit | 8-bit | ~53K | 5.0 | CRASH |
| Rapid-MLX | MLX 4-bit | 4-bit | ~83K | 4.6 | CRASH |
| llama.cpp | GGUF IQ3_XXS | Q8 | ~128K | 24.9 | 16.0 |

On 36 GB, 128K only works with llama.cpp + IQ3_XXS GGUF. MLX runtimes crash with Metal OOM around 60–100K.

#### Context ceilings (Mac mini M4, 16 GB)

| Model | Config | Usable context |
|-------|--------|---------------|
| Qwen 3.5 4B MLX 4-bit | tuned | ~28K clean |
| Qwen 3.5 9B MLX 4-bit | default flags | ~23K |
| Qwen 3.5 9B MLX 4-bit | `--prompt-cache-bytes 2GB --prefill-step-size 1024` | ~56K |
| Gemma 4 26B | UD-3bit | ~150 tokens (OOM beyond) |

Metal OOM is the real context limit. Published "131K context" does not mean your hardware can run it.

### Flag impact (M3 Max, llama.cpp, IQ4_XS, 800-token gen)

| Config | Gen tok/s | Prompt tok/s | CPU max | GPU max | Power |
|--------|-----------|-------------|---------|---------|-------|
| GPU `-ngl 99`, no LPM | 43.5 | 846 | 56°C | 95°C | 42 W GPU |
| GPU `-ngl 99` + LPM | 33.8 | 173 | 56°C | 65.8°C | 10 W GPU |
| CPU `-t 4` + LPM | 15.0 | 22 | 72.3°C | 62°C | 10 W CPU |
| `-ngl 30` partial + LPM | 18.5 | 30 | — | — | — |
| Expert offload + LPM | 14.1 | 30 | — | — | — |

---

## Thermal management (M3 Max)

Qwen 3.6 35B-A3B on M3 Max with `--n-gpu-layers 99` pushes GPU to 95°C and 42 W. The laptop becomes a space heater, fans sound like a hairdryer.

**Fix: Low Power Mode.**

```bash
sudo pmset -a lowpowermode 1
```

Verify:

```bash
pmset -g custom | grep powermode
# powermode 1  (on both Battery and AC sections)
```

Low Power Mode on Apple Silicon applies to AC power too (since macOS 12). It is the closest thing to `nvidia-smi -pl` for M-chips.

With LPM: GPU peaks at 65.8°C, 10 W. Generation speed drops only 22% because LLM inference is memory-bandwidth bound, not compute-bound — the GPU was burning watts on arithmetic it didn't need to do faster.

**GPU+LPM beats CPU at everything.** It's faster (33.8 vs 15 tok/s), CPU stays cooler (56 vs 72°C), and GPU stays under throttle.

**Don't chase CPU-only on Apple Silicon thinking it will be cooler.** GPU+LPM wins on both axes.

**Partial offloads are an anti-pattern on unified memory** for models this size. Either commit to GPU or commit to CPU. The ping-pong between CPU and GPU layers costs more than it saves.

---

## Recommended configs

### M3 Max 36 GB

| Use case | Setup |
|----------|-------|
| Daily driver, short prompts, quiet laptop | llama.cpp IQ4_XS + LPM (33.8 tok/s, 65.8°C, 32K ctx) |
| Raw speed, short prompts, don't care about heat | Rapid-MLX 4-bit, no LPM (62.3 tok/s, 76°C) |
| Long context (64K+) | llama.cpp IQ3_XXS + LPM (16–44 tok/s, 128K ctx) |
| Multi-model + admin UI | oMLX (30.6 tok/s with LPM, max_tokens honored) |

### Mac mini M4 16 GB

| Use case | Setup |
|----------|-------|
| Fast + small | Gemma 4 E2B MLX 4-bit (64.5 tok/s, 3.4 GB) |
| Best balance | Qwen 3.5 4B MLX 4-bit (38 tok/s, 2.9 GB) |
| Quality + speed tradeoff | Qwen 3.5 9B MLX 4-bit (21 tok/s, 5.6 GB) |
| Pushing limits | Qwen 3.5 35B MoE llama.cpp CPU+mmap (16.5 tok/s, 12 GB, swaps) |

---

## What local models are good for

Honestly — **not for complex agentic work**. They hallucinate on long reasoning chains and their tool-calling is unreliable.

**Good fit:**
- Classification (spam/not spam, category, priority)
- Email sorting and labeling
- Yes/no decisions
- Short summaries
- Simple extraction (get the date from this text)
- Short Q&A where the answer is already in the prompt
- Pre-processing / pre-filtering before calling a bigger cloud model

**Bad fit:**
- Multi-step coding agents
- Long reasoning chains
- Anything requiring reliable tool use across many turns
- Tasks needing broad world knowledge

---

## What didn't work

### M3 Max

- **Partial GPU offload (`-ngl 30`)** — 18.5 tok/s, worse than both pure-CPU and pure-GPU. Transfer overhead eats the savings.
- **Expert offloading via `--override-tensor`** — works great on 16 GB Mac mini, loses value on 36 GB. Speed dropped to 14.1 tok/s.
- **CPU-only inference** — 15 tok/s, CPU hits 72°C. Slower and hotter than GPU+LPM.

### Mac mini M4

- **Gemma 4 26B UD-3bit** — 31.2 tok/s gen, but OOM on prompts >150 tokens. Unusable for agents.
- **flash-moe** (Rust MoE split engine) — 9.1 tok/s on 35B vs llama.cpp's 16.5. Needs Xcode (~20 GB) + Metal Toolchain + 2-pass cargo build.
- **mlx-flash** (Python version) — OOMs before inference. `mlx_lm.load()` requires full model in RAM.
- **`-ngl 99` on Qwen 35B** — instant Metal OOM. Use `-ngl 0 --mmap`.
- **Speculative decoding with Qwen** — known mlx_lm 0.31.2 bug.
- **`llmfit` estimates** — estimated Qwen 35B at 130 tok/s, reality was 9–16. ~10× optimistic. Trust only measured numbers.

---

## Key facts

- **MLX > llama.cpp** for models <14B on Apple Silicon (~25% faster gen)
- **llama.cpp > MLX** for long context — GGUF out-scales MLX by 3–5× via mmap
- **llama.cpp > MLX** for prompt eval on cold start
- **Memory bandwidth is the hard ceiling**: M4 base ~120 GB/s, M4 Pro 273, M4 Max 410. Doubles the bandwidth → doubles the speed.
- **Metal OOM is the real context limit.** Published "131K context" ≠ your hardware can run it.
- **Day-to-day variance is real.** Same model, same Mac, different days = different tok/s. Page cache warmth matters.
- **Picking a smaller model beats any flag tuning.** 4B is 2× faster than 9B.
- **4-bit MLX = same speed as 3-bit MLX on M3 Max.** Use 4-bit for quality.
- **`mlx_lm.generate` lies about prompt tok/s** — includes cold model load time. Only `mlx_lm.server` gives real numbers.
- **`macmon` should be in your toolkit.** `brew install macmon`. Real temps, real power, no sudo.
- **LPM costs 22–33% of gen speed but drops GPU power 57–76%.**

---

## Setup & commands

### Install

```bash
brew install llama.cpp mlx_lm macmon
```

### MLX server (recommended for <14B models)

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.5-9B-4bit \
  --host 127.0.0.1 --port 8082 \
  --prompt-cache-bytes 2147483648 \
  --prefill-step-size 1024
```

OpenAI-compatible at `/v1/chat/completions`.

### llama.cpp server — small model, Metal offload

```bash
llama-server -m model.gguf --port 8080 -ngl 99 -c 131072 -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0
```

### llama.cpp server — big model, CPU + mmap (16 GB)

```bash
llama-server -m Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf --port 8081 \
  --ctx-size 16384 --n-gpu-layers 0 --mmap --flash-attn on --threads 8
```

### llama.cpp server — M3 Max daily driver with LPM

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="$HOME/Downloads/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
PORT=8081
LOG="$HOME/logs/llama-server.log"
mkdir -p "$HOME/logs"

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Server already running on port $PORT"
  exit 0
fi

sudo pmset -a lowpowermode 1 >/dev/null || true

nohup llama-server \
  --model "$MODEL" \
  --alias qwen3.6-35b-a3b \
  --port "$PORT" \
  --ctx-size 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --no-warmup \
  --jinja \
  > "$LOG" 2>&1 &

echo "llama-server starting (pid $!)"
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 2; done
echo "Ready on http://127.0.0.1:$PORT"
```

Key flags:
- **`--no-warmup`** — skips synthetic warmup at startup, keeps resident memory lower
- **`--ctx-size 32768`** — 128K KV cache costs ~3 GB even when empty. 32K covers most tasks.
- **`--cache-type-k/v q8_0`** — quantized KV cache, 2× memory savings, no quality loss

### Rapid-MLX server — Qwen 3.6 35B (M3 Max)

Two ready-to-run scripts included:

```bash
./serve-thinking.sh
```

```bash
./serve-no-thinking.sh
```

`serve-thinking.sh` — Qwen3 reasoning enabled (default `.ExtendedThinking` block, uses more tokens).

`serve-no-thinking.sh` — reasoning disabled via `--no-thinking`, faster responses, lower token usage.

Both use 4-bit KV cache quantization and tool-calling support (qwen3_xml parser). Listens on port 8081, OpenAI-compatible.

### Wiring to agent CLIs

**`.pi` agent:**

`~/.pi/agent/models.json`:
```json
{
  "llamacpp": {
    "api": "openai-completions",
    "apiKey": "llamacpp",
    "baseUrl": "http://127.0.0.1:8081/v1",
    "models": [{
      "contextWindow": 32768,
      "id": "qwen3.6-35b-a3b",
      "input": ["text"],
      "reasoning": true
    }]
  }
}
```

`~/.pi/agent/settings.json`:
```json
{
  "defaultModel": "qwen3.6-35b-a3b",
  "defaultProvider": "llamacpp"
}
```

**Hermes agent:**

`~/.hermes/config.yaml`:
```yaml
model:
  default: "mlx-community/Qwen3.5-9B-4bit"
  provider: "custom"
  base_url: "http://localhost:8082/v1"
  context_length: 32768
```

Note: Hermes `provider: "llamacpp"` silently falls back to OpenRouter. Use `"custom"`.

---

## Tested models

| HF repo | Size | Works on |
|---------|------|----------|
| `mlx-community/Qwen3.5-4B-4bit` | 2.9 GB | 16 GB, 36 GB |
| `mlx-community/Qwen3.5-9B-4bit` | 5.6 GB | 16 GB, 36 GB |
| `unsloth/Qwen3.5-9B-GGUF` Q4_K_M | 5.3 GB | 16 GB, 36 GB |
| `mlx-community/gemma-4-e2b-it-4bit` | 3.4 GB | 16 GB, 36 GB |
| `mlx-community/gemma-4-e4b-it-8bit` | 8.4 GB | 36 GB |
| `unsloth/gemma-4-26b-a4b-it-UD-MLX-3bit` | 12 GB | 16 GB (limited), 36 GB |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | 19 GB | 36 GB |
| `unsloth/Qwen3.5-35B-A3B-GGUF` UD-IQ3_XXS | 12 GB | 16 GB (CPU), 36 GB |
| `mlx-community/Phi-4-mini-reasoning-4bit` | 2.0 GB | 16 GB, 36 GB |
| `bourn23/nvidia-llama-3.1-nemotron-nano-8b-v1-mlx-4bit` | 4.2 GB | 16 GB, 36 GB |

---

## Sources

- [jock.pl — 35B on Mac mini](https://thoughts.jock.pl/p/local-llm-35b-mac-mini-gemma-swap-production-2026)
- [Hermes Mac guide](https://hermes-agent.nousresearch.com/docs/guides/local-llm-on-mac)
- [flash-moe](https://github.com/philtrem/qwen3.5-gemma4-moe-flash-mlx-turbo-quant)
- [mlx-flash](https://github.com/Blaiza/mlx-flash)
- [llmfit](https://github.com/nasser-ali-ahmed/llmfit)
- [BleedingDev/local-llm-extreme-experiments](https://github.com/BleedingDev/local-llm-extreme-experiments)
- [llama.cpp Apple Silicon perf](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [MLX vs llama.cpp Qwen 3](https://github.com/ggml-org/llama.cpp/issues/19366)
- [macmon](https://github.com/vladkens/macmon)
