// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: printf '0 0\n# A comment and an empty line.\n\n0 1\n  256\t32  \n0 64\n' > %t.testdir/batch.txt
// RUN: ls %t.testdir/*.json | xargs -I {} %omp-kernel-replay --verify --repetitions=3 --batch %t.testdir/batch.txt {} | FileCheck --check-prefix=BATCH %s
// RUN: printf '0 1\n2 32\n0 64\n' > %t.testdir/batch-limits.txt
// RUN: ls %t.testdir/*.json | xargs -I {} not %omp-kernel-replay --verify --batch %t.testdir/batch-limits.txt {} > %t.testdir/limits.out 2> %t.testdir/limits.err
// RUN: FileCheck --check-prefix=LIMITS-OUT %s < %t.testdir/limits.out
// RUN: FileCheck --check-prefix=LIMITS-ERR %s < %t.testdir/limits.err
// RUN: printf '1 1\n2\n' > %t.testdir/batch-invalid.txt
// RUN: ls %t.testdir/*.json | xargs -I {} not %omp-kernel-replay --batch %t.testdir/batch-invalid.txt {} 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: ls %t.testdir/*.json | xargs -I {} not %omp-kernel-replay --num-teams=2 --batch %t.testdir/batch.txt {} 2>&1 | FileCheck --check-prefix=EXCLUSIVE %s
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// BATCH:      ||ReplayConfig|| index: 0 num_teams: 256 num_threads: {{[1-9][0-9]*}}
// BATCH-COUNT-3: Replay time
// BATCH-NEXT: ||ReplayTiming|| duration_seconds: {{[0-9.e+-]+}}
// BATCH-NEXT: Replay done, device memory verified
// BATCH-NEXT: ||ReplayConfig|| index: 1 num_teams: 256 num_threads: 1
// BATCH-COUNT-3: Replay time
// BATCH-NEXT: ||ReplayTiming||
// BATCH-NEXT: Replay done, device memory verified
// BATCH-NEXT: ||ReplayConfig|| index: 2 num_teams: 256 num_threads: 32
// BATCH-COUNT-3: Replay time
// BATCH-NEXT: ||ReplayTiming||
// BATCH-NEXT: Replay done, device memory verified
// BATCH-NEXT: ||ReplayConfig|| index: 3 num_teams: 256 num_threads: 64
// BATCH-COUNT-3: Replay time
// BATCH-NEXT: ||ReplayTiming||
// BATCH-NEXT: Replay done, device memory verified
// BATCH-NOT: ||ReplayConfig||

// LIMITS-OUT:      ||ReplayConfig|| index: 0 num_teams: 256 num_threads: 1
// LIMITS-OUT:      Replay done, device memory verified
// LIMITS-OUT-NEXT: ||ReplayConfig|| index: 1 num_teams: 2 num_threads: 32
// LIMITS-OUT-NEXT: ||ReplayError|| index: 1
// LIMITS-OUT-NEXT: ||ReplayConfig|| index: 2 num_teams: 256 num_threads: 64
// LIMITS-OUT:      Replay done, device memory verified
// LIMITS-ERR: [llvm-omp-kernel-replay] Error: configuration 1: number of teams (2) is out of the allowed limits (min,max: {{[0-9]+}},{{[0-9]+}})
// LIMITS-ERR: [llvm-omp-kernel-replay] Error: 1 of 3 batch configurations failed

// INVALID: [llvm-omp-kernel-replay] Error: invalid batch file line 2: expected '<num-teams> <num-threads>'
// EXCLUSIVE: [llvm-omp-kernel-replay] Error: --batch cannot be combined with --num-teams or --num-threads

#include <cstdint>
#include <cstdio>

int main() {
  size_t Size = 1000;
  uint64_t *Data = new uint64_t[Size];

  for (size_t I = 0; I < Size; ++I) {
    Data[I] = 20;
  }

#pragma omp target teams distribute parallel for num_teams(256)                \
    thread_limit(128) map(tofrom : Data[0 : Size])
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = 10 + (uint64_t)I;
  }

  uint64_t Sum = 0;
  for (size_t I = 0; I < Size; ++I) {
    Sum += Data[I];
  }

  // CHECK: PASS
  if (Sum == 509500)
    printf("PASS\n");

  delete[] Data;
}
