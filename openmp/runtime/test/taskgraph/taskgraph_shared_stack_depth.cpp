// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// XFAIL: *
// clang-format on

#include <cstdio>

static volatile int StackSink = 0;

struct Payload {
  int values[6];
  int bias;
};

__attribute__((noinline)) static int evaluate_payload(const Payload &payload,
                                                      int seed) {
  return seed * payload.values[0] - payload.values[1] +
         payload.values[2] * payload.values[3] - payload.bias +
         payload.values[4] - payload.values[5];
}

__attribute__((noinline)) static void clobber_stack(int base) {
  volatile int scratch[4096];

  for (int i = 0; i < 4096; ++i)
    scratch[i] = base + i;

  StackSink += scratch[base & 63];
}

__attribute__((noinline)) static int run_taskgraph(int seed) {
  Payload payload{
      {seed + 1, seed + 3, seed + 5, seed + 7, seed + 11, seed + 13},
      seed * 17 + 19};
  int result = -1;

#pragma omp taskgraph graph_id(92)
  {
#pragma omp task firstprivate(payload, seed) shared(result)
    {
      result = evaluate_payload(payload, seed);
    }
  }

  return result;
}

__attribute__((noinline)) static int call_with_depth(int seed, int depth) {
  volatile int padding[128];

  for (int i = 0; i < 128; ++i)
    padding[i] = seed + depth + i;

  StackSink += padding[(seed + depth) & 127];

  if (depth == 0)
    return run_taskgraph(seed);
  return call_with_depth(seed, depth - 1);
}

__attribute__((noinline)) static int expected_result(int seed) {
  Payload payload{
      {seed + 1, seed + 3, seed + 5, seed + 7, seed + 11, seed + 13},
      seed * 17 + 19};
  return evaluate_payload(payload, seed);
}

int main() {
  constexpr int RecordSeed = 3;
  constexpr int ReplaySeed = 17;

  const int recorded = call_with_depth(RecordSeed, 0);
  if (recorded != expected_result(RecordSeed)) {
    std::fprintf(stderr, "FAIL initial record got=%d expected=%d\n", recorded,
                 expected_result(RecordSeed));
    return 1;
  }

  clobber_stack(ReplaySeed * 1000);
  const int replayed = call_with_depth(ReplaySeed, 3);
  if (replayed != recorded) {
    std::fprintf(
        stderr, "BUG shared stack replay depth=%d seed=%d got=%d expected=%d\n",
        3, ReplaySeed, replayed, recorded);
    return 1;
  }

  std::fprintf(stderr, "PASS shared stack replay=%d\n", replayed);
  return 0;
}

// CHECK: PASS shared stack replay=14
