// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// Minimal end-to-end exercise of reduction(+: ...) on a replayable taskloop
// that is dynamically (non-lexically) nested inside a taskgraph: the
// taskloop directive lives in a helper called from inside the taskgraph
// region, not in the taskgraph's lexical body.
//
//   - Sum is a file-scope static, so the reduction's accumulator address is
//     stable across runs and the taskgraph relocate helper has no shareds
//     slot it must refresh (the static capture is link-time-fixed).
//   - 'seed' is captured into the taskloop's per-task '.kmp_privates.t'
//     snapshot as firstprivate(saved: ...); per OpenMP 6.0 [7.5.4] / [14.3]
//     the snapshot is taken at recording time and reused unchanged on every
//     replay.
//   - The recording run therefore produces the expected reduction for
//     Seeds[0], and every subsequent replay (each invoked with a different
//     seed) produces exactly the recorded value because the saved snapshot
//     of 'seed' is what the body sees.
//
// Historically this test crashed on replay with OMP: Error #302 because
// the taskgraph relocate helper refused to handle the (static) reduction
// capture, and even after that was fixed the reduction body still
// asserted because the recording-time taskgroup_t had been torn down and
// the taskred state was unreachable.  Both gaps are addressed: the
// relocate helper now treats captures of static-storage variables as
// no-op-safe, and the runtime now stashes reduction-init input into the
// surrounding taskgraph at recording so the replay machinery can re-create
// the taskred state on every replay.

#include <cstdio>

static volatile int Sum = 0;

__attribute__((noinline)) static void emit_reduction_taskloop(int seed) {
#pragma omp taskloop replayable num_tasks(8) reduction(+ : Sum)                \
    firstprivate(saved : seed)
  for (int i = 0; i < 16; ++i)
    Sum += seed + i;
}

__attribute__((noinline)) static int run_taskgraph(int seed) {
  Sum = 0;

#pragma omp taskgraph graph_id(921)
  {
    emit_reduction_taskloop(seed);
  }

  return Sum;
}

__attribute__((noinline)) static int expected_result(int seed) {
  int sum = 0;
  for (int i = 0; i < 16; ++i)
    sum += seed + i;
  return sum;
}

int main() {
  constexpr int NumRuns = 4;
  constexpr int Seeds[NumRuns] = {1, 5, 11, 23};

  int recorded = -1;
  bool failed = false;

#pragma omp parallel num_threads(4)
  {
#pragma omp single
    {
      recorded = run_taskgraph(Seeds[0]);
      const int exp0 = expected_result(Seeds[0]);
      if (recorded != exp0) {
        std::fprintf(stderr, "FAIL initial record got=%d expected=%d\n",
                     recorded, exp0);
        failed = true;
      }

      // 'seed' is firstprivate(saved:), so every replay's body sees the
      // snapshot taken at recording time.  The reduction therefore yields
      // the same value as the recording regardless of the live argument
      // passed on the replay.
      for (int i = 1; i < NumRuns; ++i) {
        const int replayed = run_taskgraph(Seeds[i]);
        if (replayed != recorded) {
          std::fprintf(stderr, "FAIL replay %d seed=%d got=%d recorded=%d\n", i,
                       Seeds[i], replayed, recorded);
          failed = true;
        }
      }
    }
  }

  if (failed)
    return 1;

  std::fprintf(stderr,
               "PASS non-lexical replayable taskloop reduction result=%d\n",
               recorded);
  return 0;
}

// CHECK: PASS non-lexical replayable taskloop reduction result=
