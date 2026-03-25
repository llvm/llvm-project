// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t
// RUN: env OMP_NUM_THREADS=4 %libomp-run 2>&1 | FileCheck %s
// REQUIRES: omp_taskgraph_experimental
// clang-format on

// A taskloop reduction's result is only combined into the original variable
// when the taskloop's implicit taskgroup ends.  On replay, a task that depends
// on the reduction must therefore not start until that combine has happened,
// not merely once the reduction's member tasks have finished.  The second
// taskloop reads 'res' through a shared reference; if it is released as soon
// as the first taskloop's chunks complete it sees 0 instead of 45.

#include <cstdio>

int main() {
  int arr[10], arr2[10];
  for (int i = 0; i < 10; i++) {
    arr[i] = i;
    arr2[i] = 3 + i * 2;
  }

  int bad = 0;
#pragma omp parallel
#pragma omp single
  for (int i = 0; i < 4; i++) {
    int res = 0, res2 = 0, seen = -1;
#pragma omp taskgraph graph_id(9107)
    {
#pragma omp taskloop reduction(+ : res) num_tasks(10)
      for (int j = 0; j < 10; j++)
        res += arr[j];
#pragma omp taskloop reduction(+ : res2) num_tasks(10) shared(res, seen)
      for (int j = 0; j < 10; j++) {
        res2 += res * arr2[j];
        if (j == 0)
          seen = res;
      }
    }
    if (res != 45 || res2 != 5400 || seen != 45) {
      std::fprintf(stderr, "FAIL iter %d res=%d res2=%d seen=%d\n", i, res,
                   res2, seen);
      bad = 1;
    }
  }

  if (bad)
    return 1;
  std::fprintf(stderr, "PASS reduction join\n");
  return 0;
}

// CHECK: PASS reduction join
