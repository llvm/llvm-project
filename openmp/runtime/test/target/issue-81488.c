// RUN: %libomp-compile
// RUN: env OMP_NUM_THREADS=1 LIBOMP_USE_HIDDEN_HELPER_TASK=1 \
// RUN:     LIBOMP_NUM_HIDDEN_HELPER_THREADS=8 %libomp-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define Nz 8
#define DEVICE_ID 0

int a[Nz];

int main(void) {
  for (int n = 0; n < 10; ++n) {
    for (int k = 0; k < Nz; ++k) {
      a[k] = -1;
    }
#pragma omp parallel shared(a)
    {
#pragma omp single
      {
#pragma omp target teams distribute parallel for nowait device(DEVICE_ID)      \
    map(tofrom : a[0 : 8])
        for (int i = 0; i < Nz; ++i) {
          a[i] = i;
        }
      }
#pragma omp barrier
    }
    for (int k = 0; k < Nz; ++k) {
      printf("a[%d] = %d\n", k, a[k]);
    }
  }
  return 0;
}
