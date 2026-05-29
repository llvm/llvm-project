// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

#include <cstdio>

static volatile int StackSink = 0;
// Aggregated across the taskloop body via #pragma omp atomic. Lives in
// static storage so the address never moves across replays (this test is
// about firstprivate(saved:), not reductions: reductions on replayable
// taskgraphs are tracked separately).
static volatile int Aggregate = 0;

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

// The taskloop directive lives in a helper function called from inside the
// taskgraph region, so the loop's firstprivate(saved:) captures originate
// in a non-lexical surrounding scope. The taskloop relocation helper has
// nothing to refresh at replay since the saved snapshots are sourced from
// '.kmp_privates.t', not the shareds slots.
__attribute__((noinline)) static void emit_replayable_taskloop(int seed) {
  Payload payload{{seed + 1, seed + 3, seed + 5, seed + 7, seed + 11,
                   seed + 13},
                  seed * 17 + 19};

#pragma omp taskloop replayable num_tasks(8) firstprivate(saved : payload, seed)
  for (int i = 0; i < 16; ++i) {
    int contribution = evaluate_payload(payload, seed + i);
#pragma omp atomic
    Aggregate += contribution;
  }
}

__attribute__((noinline)) static int run_taskgraph(int seed) {
  Aggregate = 0;

#pragma omp taskgraph graph_id(633)
  { emit_replayable_taskloop(seed); }

  return Aggregate;
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
  Payload payload{{seed + 1, seed + 3, seed + 5, seed + 7, seed + 11,
                   seed + 13},
                  seed * 17 + 19};
  int sum = 0;
  for (int i = 0; i < 16; ++i)
    sum += evaluate_payload(payload, seed + i);
  return sum;
}

int main() {
  constexpr int NumCalls = 4;
  constexpr int Seeds[NumCalls] = {3, 17, 29, 41};
  constexpr int Depths[NumCalls] = {0, 3, 1, 5};

  int recorded = -1;
  bool failed = false;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      recorded = call_with_depth(Seeds[0], Depths[0]);
      if (recorded != expected_result(Seeds[0])) {
        std::fprintf(stderr,
                     "FAIL initial taskloop record got=%d expected=%d\n",
                     recorded, expected_result(Seeds[0]));
        failed = true;
      }

      for (int i = 1; i < NumCalls; ++i) {
        clobber_stack(Seeds[i] * 1000);
        const int replayed = call_with_depth(Seeds[i], Depths[i]);
        if (replayed != recorded) {
          std::fprintf(stderr,
                       "FAIL taskloop replay %d depth=%d seed=%d got=%d expected=%d\n",
                       i, Depths[i], Seeds[i], replayed, recorded);
          failed = true;
        }
      }
    }
  }

  if (failed)
    return 1;

  std::fprintf(stderr, "PASS replayable taskloop saved stack result=%d\n", recorded);
  return 0;
}

// CHECK: PASS replayable taskloop saved stack result=
