#include <omp.h>
#include <stdio.h>

// RUN: %libomptarget-compile-amdgcn-amd-amdhsa -O2 -mcode-object-version=5

// RUN: env OMP_TARGET_OFFLOAD=MANDATORY \
// RUN: env LIBOMPTARGET_STACK_SIZE=4096 \
// RUN:   %libomptarget-run-amdgcn-amd-amdhsa 2>&1 \
// RUN:   | %fcheck-amdgcn-amd-amdhsa

// RUN: env OMP_TARGET_OFFLOAD=MANDATORY \
// RUN: env LIBOMPTARGET_STACK_SIZE=131073 \
// RUN:   %libomptarget-run-amdgcn-amd-amdhsa 2>&1 \
// RUN:   | %fcheck-amdgcn-amd-amdhsa -check-prefix=LIMIT_EXCEEDED

// TODO: Realize the following run in an acceptable manner.
//       Unfortunately with insufficient scratch mem size programs will hang.
//       Therefore, a timeout mechanism would help tremendously.
//       Additionally, we need to allow empty output / unsuccessful execution.

// RUN?: env OMP_TARGET_OFFLOAD=MANDATORY \
// RUN?: env LIBOMPTARGET_STACK_SIZE=16 \
// RUN?:   timeout 10 %libomptarget-run-amdgcn-amd-amdhsa 2>&1 \
// RUN?:   | %fcheck-amdgcn-amd-amdhsa -check-prefix=LIMIT_INSUFFICIENT \
// RUN?:   --allow-empty

// REQUIRES: amdgcn-amd-amdhsa

// Cause the compiler to set amdhsa_uses_dynamic_stack to '1' using recursion.
// That is: stack requirement for main's target region may not be calculated.

// This recursive function will eventually return 0.
int recursiveFunc(const int Recursions) {
  if (Recursions < 1)
    return 0;

  int j[Recursions];
#pragma omp target private(j)
  { ; }

  return recursiveFunc(Recursions - 1);
}

int main() {
  int N = 256;
  int a[N];
  int b[N];
  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j] + recursiveFunc(j);
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: Success

/// LIMIT_EXCEEDED: Scratch memory size will be set to
/// LIMIT_EXCEEDED: Success

/// LIMIT_INSUFFICIENT-NOT: Success
