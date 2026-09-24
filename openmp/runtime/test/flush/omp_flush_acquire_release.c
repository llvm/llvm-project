// RUN: %libomp-compile-and-run
// REQUIRES: clang

// Test is based on OpenMP API Example (omp_5.0) acquire_release3.c
// https://github.com/OpenMP/Examples/blob/main/synchronization/sources/acquire_release.3.c

#include <stdio.h>
#include <omp.h>

typedef void ident_t;
extern void __kmpc_flush_explicit(ident_t *, int order, int scope);

int test_memorder(int write_order, int read_order) {
  int x = 0, y = 0;
  int num_fails = 0;
#pragma omp parallel num_threads(2)
  {
    int thrd = omp_get_thread_num();
    if (thrd == 0) {
      x = 10;
      __kmpc_flush_explicit(NULL, write_order, 0);
#pragma omp atomic write // or with relaxed clause
      y = 1;
    } else {
      int tmp = 0;
      while (tmp == 0) {
#pragma omp atomic read // or with relaxed clause
        tmp = y;
      }
      __kmpc_flush_explicit(NULL, read_order, 0);
      // printf("x = %d\n", x);  // always "x = 10"
      if (x != 10)
        num_fails++;
    }
  }
  return num_fails;
}

int main() {
  // Clang-based compiler has predefined macro __ATOMIC_<memory_order>.
  int write_order[3] = {__ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL, __ATOMIC_RELEASE};
  int read_order[3] = {__ATOMIC_SEQ_CST, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE};

  // Repeat 1000 times
  for (int n = 0; n < 1000; n++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (test_memorder(write_order[i], read_order[j])) {
          printf("failed\n");
          exit(EXIT_FAILURE);
        }
      }
    }
  }

  printf("passed\n");
  return EXIT_SUCCESS;
}
