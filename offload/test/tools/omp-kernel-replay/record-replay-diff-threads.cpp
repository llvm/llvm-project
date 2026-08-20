// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir LIBOMPTARGET_RECORD_REPORT_FILENAME=report.txt %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: awk '/\.json/ {print $1}' %t.testdir/report.txt | tr -d ',' > %t.testdir/json_list.txt
// RUN: cat %t.testdir/json_list.txt | count 1
// RUN: ls -1 %t.testdir/*.json | count 1
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=1 %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=32 %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=64 %t.testdir/{}
// clang-format on

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

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
