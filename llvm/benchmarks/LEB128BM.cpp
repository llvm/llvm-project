#include "benchmark/benchmark.h"
#include "llvm/Support/LEB128.h"

#include <cstdint>
#include <vector>

using namespace llvm;

static std::vector<uint8_t>
encodeManyULEB128(const std::vector<uint64_t> &Vals) {
  std::vector<uint8_t> Buf;
  Buf.resize(Vals.size() * 10);
  uint8_t *P = Buf.data();
  for (uint64_t V : Vals)
    P += encodeULEB128(V, P);
  Buf.resize(P - Buf.data());
  return Buf;
}

static std::vector<uint8_t>
encodeManySLEB128(const std::vector<int64_t> &Vals) {
  std::vector<uint8_t> Buf;
  Buf.resize(Vals.size() * 10);
  uint8_t *P = Buf.data();
  for (int64_t V : Vals)
    P += encodeSLEB128(V, P);
  Buf.resize(P - Buf.data());
  return Buf;
}

// Decode a stream of ULEB128 values where all values fit in one byte (0-127).
static void BM_DecodeULEB128_1Byte(benchmark::State &State) {
  std::vector<uint64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = I & 0x7f;
  auto Encoded = encodeManyULEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeULEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Decode a stream of ULEB128 values needing 2 bytes (128-16383).
static void BM_DecodeULEB128_2Byte(benchmark::State &State) {
  std::vector<uint64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = 128 + (I & 0x3fff);
  auto Encoded = encodeManyULEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeULEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Decode a stream of ULEB128 values needing 5 bytes.
static void BM_DecodeULEB128_5Byte(benchmark::State &State) {
  std::vector<uint64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = (1ULL << 28) + I;
  auto Encoded = encodeManyULEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeULEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Decode a stream of ULEB128 values needing 10 bytes (max for uint64_t).
static void BM_DecodeULEB128_10Byte(benchmark::State &State) {
  std::vector<uint64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = UINT64_MAX - I;
  auto Encoded = encodeManyULEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeULEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Decode a stream of SLEB128 values where all values fit in one byte (-64..63).
static void BM_DecodeSLEB128_1Byte(benchmark::State &State) {
  std::vector<int64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = (int64_t)(I & 0x7f) - 64;
  auto Encoded = encodeManySLEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeSLEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Decode a stream of SLEB128 values needing 2 bytes.
static void BM_DecodeSLEB128_2Byte(benchmark::State &State) {
  std::vector<int64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I)
    Vals[I] = (int64_t)(I & 0x1fff) - 4096;
  auto Encoded = encodeManySLEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeSLEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

// Mixed: 70% 1-byte, 20% 2-byte, 10% 5-byte (realistic DWARF-like
// distribution).
static void BM_DecodeULEB128_Mixed(benchmark::State &State) {
  std::vector<uint64_t> Vals(4096);
  for (size_t I = 0; I < Vals.size(); ++I) {
    unsigned Bucket = I % 10;
    if (Bucket < 7)
      Vals[I] = I & 0x7f;
    else if (Bucket < 9)
      Vals[I] = 128 + (I & 0x3fff);
    else
      Vals[I] = (1ULL << 28) + I;
  }
  auto Encoded = encodeManyULEB128(Vals);

  for (auto _ : State) {
    const uint8_t *P = Encoded.data();
    const uint8_t *End = P + Encoded.size();
    while (P < End) {
      unsigned N;
      benchmark::DoNotOptimize(decodeULEB128(P, &N, End));
      P += N;
    }
  }
  State.SetBytesProcessed(State.iterations() * Encoded.size());
}

BENCHMARK(BM_DecodeULEB128_1Byte);
BENCHMARK(BM_DecodeULEB128_2Byte);
BENCHMARK(BM_DecodeULEB128_5Byte);
BENCHMARK(BM_DecodeULEB128_10Byte);
BENCHMARK(BM_DecodeSLEB128_1Byte);
BENCHMARK(BM_DecodeSLEB128_2Byte);
BENCHMARK(BM_DecodeULEB128_Mixed);

BENCHMARK_MAIN();
