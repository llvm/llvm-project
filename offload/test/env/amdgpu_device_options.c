// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DATA
// RUN: env LIBOMPTARGET_DEBUG=1 LIBOMPTARGET_AMDGPU_USE_MULTIPLE_SDMA_ENGINES=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOSDMA
// REQUIRES: amdgpu
// REQUIRES: libomptarget-debug
// clang-format on

#include <stdio.h>

// Check both copy paths.
#define SMALL 256
#define LARGE (1 << 20)

static int Small[SMALL];
static int Large[LARGE];

int main(void) {
  for (int I = 0; I < SMALL; ++I)
    Small[I] = I;
  for (int I = 0; I < LARGE; ++I)
    Large[I] = I;

#pragma omp target teams distribute parallel for map(tofrom : Small, Large)
  for (int I = 0; I < LARGE; ++I) {
    if (I < SMALL)
      Small[I] += 1;
    Large[I] += 1;
  }

  int Errors = 0;
  for (int I = 0; I < SMALL; ++I)
    Errors += Small[I] != I + 1;
  for (int I = 0; I < LARGE; ++I)
    Errors += Large[I] != I + 1;

  printf("Errors = %d\n", Errors);
  return Errors != 0;
}

// DATA: Errors = 0
// NOSDMA-NOT: Running Async Copy on SDMA Engine
// NOSDMA: Errors = 0
