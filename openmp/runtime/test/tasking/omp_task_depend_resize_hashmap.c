// RUN: %libomp-compile && env KMP_ENABLE_TASK_THROTTLING=0 %libomp-run

// This test is known to be fragile on NetBSD kernel at the moment,
// https://bugs.llvm.org/show_bug.cgi?id=42020.
// UNSUPPORTED: netbsd
#include<omp.h>
#include<stdlib.h>
#include<string.h>

// The first hashtable static size is 997
#define NUM_DEPS 4000


int main()
{
  int *deps = calloc(NUM_DEPS, sizeof(int));
  int i;
  int failed = 0;

  #pragma omp parallel
  #pragma omp master
  {
    for (i = 0; i < NUM_DEPS; i++) {
      #pragma omp task firstprivate(i) depend(inout: deps[i])
      {
        deps[i] = 1;
      }
      #pragma omp task firstprivate(i) depend(inout: deps[i])
      {
        deps[i] = 2;
      }
    }
  }

  for (i = 0; i < NUM_DEPS; i++) {
    if (deps[i] != 2)
      failed++;
  }

  return failed;
}
