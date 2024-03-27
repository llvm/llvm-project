// RUN: %libomp-compile-and-run
// RUN: %libomp-compile && env OMP_NUM_THREADS='1' %libomp-run

/*
https://github.com/llvm/llvm-project/issues/81488

Assertion failure at kmp_runtime.cpp(2460): master_th->th.th_task_team ==
team->t.t_task_team[master_th->th.th_task_state].
OMP: Error #13: Assertion failure at kmp_runtime.cpp(2460).

The assertion fails with OMP_NUM_THREADS=1.

*/

#include <omp.h>

#define Nz 8
#define DEVICE_ID 0

int main(void) {
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp target teams distribute parallel for nowait device(DEVICE_ID)
      for (int i = 0; i < Nz; ++i) {
      }
    }
#pragma omp barrier
  }
  return 0;
}
