// RUN: %clang %openmp_flags %flags-use-compiler-omp-h %s -o %t && %libomp-run
// UNSUPPORTED: gcc-5, gcc-6, gcc-7, gcc-8
// UNSUPPORTED: clang-5, clang-6, clang-7, clang-8, clang-9, clang-10
// UNSUPPORTED: icc

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_my_sleep.h"

int a, b;

void mutexinoutset_task() {
  if (b != 0) {
    fprintf(stderr, "mutexinoutset_task: b != 0 at start of task\n");
    exit(EXIT_FAILURE);
  }
  b++;
  if (b != 1) {
    fprintf(stderr, "mutexinoutset_task: b != 1\n");
    exit(EXIT_FAILURE);
  }
  my_sleep(0.1);
  b--;
  if (b != 0) {
    fprintf(stderr, "mutexinoutset_task: b != 0 at end of task\n");
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  omp_depend_t dep_a_in;
  omp_depend_t dep_a_out;
  omp_depend_t dep_a_inout;
  omp_depend_t dep_a_mutexinoutset;

  a = 0;
  b = 0;

  #pragma omp depobj(dep_a_in) depend(in: a)
  #pragma omp depobj(dep_a_out) depend(out: a)
  #pragma omp depobj(dep_a_inout) depend(inout: a)
  #pragma omp depobj(dep_a_mutexinoutset) depend(mutexinoutset: a)

  #pragma omp parallel
  {
    #pragma omp single
    {

      #pragma omp task depend(depobj: dep_a_out)
      {
        my_sleep(0.1);
        a = 10;
      }

      #pragma omp task depend(depobj: dep_a_inout)
      {
        my_sleep(0.1);
        a++;
      }

      #pragma omp task depend(depobj: dep_a_mutexinoutset)
      mutexinoutset_task();
      #pragma omp task depend(depobj: dep_a_mutexinoutset)
      mutexinoutset_task();
      #pragma omp task depend(depobj: dep_a_mutexinoutset)
      mutexinoutset_task();
      #pragma omp task depend(depobj: dep_a_mutexinoutset)
      mutexinoutset_task();
      #pragma omp task depend(depobj: dep_a_mutexinoutset)
      mutexinoutset_task();

      #pragma omp task depend(depobj: dep_a_in)
      { a += 10; }
    }
  }

  if (a != 21) {
    fprintf(stderr, "a (%d) != 21\n", a);
    exit(EXIT_FAILURE);
  }

  #pragma omp depobj(dep_a_in) destroy
  #pragma omp depobj(dep_a_out) destroy
  #pragma omp depobj(dep_a_inout) destroy
  #pragma omp depobj(dep_a_mutexinoutset) destroy

  return EXIT_SUCCESS;
}
