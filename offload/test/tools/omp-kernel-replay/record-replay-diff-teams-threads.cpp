// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir LIBOMPTARGET_RECORD_REPORT_FILENAME=report.txt %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: awk '/\.json/ {print $1}' %t.testdir/report.txt | tr -d ',' > %t.testdir/json_list.txt
// RUN: cat %t.testdir/json_list.txt | count 2
// RUN: ls -1 %t.testdir/*.json | count 2
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-teams=1 --num-threads=1 %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-teams=2 --num-threads=32 %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-teams=32 --num-threads=64 %t.testdir/{}
// RUN: sed -n '1p' %t.testdir/json_list.txt | xargs -I {} not %omp-kernel-replay --verify --num-threads=129 %t.testdir/{} 2>&1 | FileCheck --check-prefix=REPLAY-ERROR1 %s
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=127 %t.testdir/{}
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} not %omp-kernel-replay --verify --num-threads=1024 %t.testdir/{} 2>&1 | FileCheck --check-prefix=REPLAY-ERROR1 %s
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} not %omp-kernel-replay --verify --num-teams=2 %t.testdir/{} 2>&1 | FileCheck --check-prefix=REPLAY-ERROR2 %s

// RUN: %libomptarget-compilexx-generic -mllvm -openmp-ir-builder-use-default-max-threads=0
// RUN: rm -rf %t.testdir
// RUN: mkdir -p %t.testdir
// RUN: env LIBOMPTARGET_RECORD=1 LIBOMPTARGET_RECORD_MEMSIZE=536870912 LIBOMPTARGET_RECORD_DIR=%t.testdir LIBOMPTARGET_RECORD_REPORT_FILENAME=report.txt %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: awk '/\.json/ {print $1}' %t.testdir/report.txt | tr -d ',' > %t.testdir/json_list.txt
// RUN: cat %t.testdir/json_list.txt | count 2
// RUN: ls -1 %t.testdir/*.json | count 2
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=129 %t.testdir/{}
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} %omp-kernel-replay --verify --num-threads=1024 %t.testdir/{}
// RUN: sed -n '2p' %t.testdir/json_list.txt | xargs -I {} not %omp-kernel-replay --verify --num-threads=2048 %t.testdir/{} 2>&1 | FileCheck --check-prefix=REPLAY-ERROR1 %s

// REQUIRES: gpu

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: intelgpu

// REPLAY-ERROR1: [llvm-omp-kernel-replay] Error: number of threads ({{[0-9]+}}) is out of the allowed limits (min,max: 1,{{[0-9]+}})
// REPLAY-ERROR2: [llvm-omp-kernel-replay] Error: number of teams (2) is out of the allowed limits (min,max: 1,1)
// clang-format on

#include <cstdint>
#include <cstdio>

int main() {
  size_t Size = 1000;
  uint64_t *Data = new uint64_t[Size];

  for (size_t I = 0; I < Size; ++I) {
    Data[I] = 20;
  }

#pragma omp target teams distribute parallel for thread_limit(128)             \
    map(tofrom : Data[0 : Size])
  for (size_t I = 0; I < Size; ++I) {
    Data[I] = 10 + (uint64_t)I;
  }

#pragma omp target
  {
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
