// RUN: %libomp-compile-and-run
// RUN: env OMP_NUM_THREADS=1 %libomp-run

#include <omp.h>

#define Nz 8
#define DEVICE_ID 0

int main(void) {
  for (int n = 0; n < 10; ++n) {
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
  }
  return 0;
}
