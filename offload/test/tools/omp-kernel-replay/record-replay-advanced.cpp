// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: ls -t %t.testdir/*.json | sed -n '1p' | grep . | xargs -I {} %omp-kernel-replay --verify {}
// RUN: ls -t %t.testdir/*.json | sed -n '2p' | grep . | xargs -I {} %omp-kernel-replay --verify {}
// RUN: ls -t %t.testdir/*.json | sed -n '3p' | grep . | xargs -I {} %omp-kernel-replay --verify {}
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

#include <cstdint>
#include <cstdio>

#pragma omp declare target
uint64_t GlobalSum = 0;
#pragma omp end declare target

int main() {
  size_t Size = 1000;
  uint64_t *Data = new uint64_t[Size];

  for (size_t I = 0; I < Size; ++I) {
    Data[I] = 20;
  }

  uint64_t Sum = 0;

#pragma omp target data map(tofrom : Data[0 : Size])
  {
#pragma omp target teams distribute parallel for num_teams(256)                \
    thread_limit(128)
    for (size_t I = 0; I < Size; ++I) {
      Data[I] = 10 + (uint64_t)I;
    }

#pragma omp target thread_limit(1) map(tofrom : Sum)
    {
      for (size_t I = 0; I < Size; ++I) {
        GlobalSum += Data[I];
      }
      Sum = GlobalSum;
    }

    // CHECK: PASS
    if (Sum == 509500)
      printf("PASS\n");

#pragma omp target teams distribute parallel for num_teams(512) thread_limit(64)
    for (size_t I = 0; I < Size; ++I) {
      Data[I] = GlobalSum;
    }
  }

  Sum = 0;
  for (size_t I = 0; I < Size; ++I) {
    Sum += Data[I];
  }

  // CHECK: PASS
  if (Sum == 509500000ULL)
    printf("PASS\n");

  delete[] Data;
}
