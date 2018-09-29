// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

const int MaxThreads = 1024;
const int NumThreads = 64;

int main(int argc, char *argv[]) {
  int level = -1, activeLevel = -1;
  int check1[MaxThreads];
  int check2[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check1[i] = check2[i] = 0;
  }

  #pragma omp target map(level, activeLevel, check1[:], check2[:])
  {
    level = omp_get_level();
    activeLevel = omp_get_active_level();

    // Expecting active parallel region.
    #pragma omp parallel num_threads(NumThreads)
    {
      int id = omp_get_thread_num();
      // Multiply return value of omp_get_level by 5 to avoid that this test
      // passes if both API calls return wrong values.
      check1[id] += omp_get_level() * 5 + omp_get_active_level();

      // Expecting serialized parallel region.
      #pragma omp parallel
      {
        #pragma omp atomic
        check2[id] += omp_get_level() * 5 + omp_get_active_level();
      }
    }
  }

  // CHECK: target: level = 0, activeLevel = 0
  printf("target: level = %d, activeLevel = %d\n", level, activeLevel);

  // CHECK-NOT: invalid
  for (int i = 0; i < MaxThreads; i++) {
    // Check active parallel region:
    // omp_get_level() = 1, omp_get_active_level() = 1
    const int Expected1 = 6;

    if (i < NumThreads) {
      if (check1[i] != Expected1) {
        printf("invalid: check1[%d] should be %d, is %d\n", i, Expected1, check1[i]);
      }
    } else if (check1[i] != 0) {
      printf("invalid: check1[%d] should be 0, is %d\n", i, check1[i]);
    }

    // Check serialized parallel region:
    // omp_get_level() = 2, omp_get_active_level() = 1
    const int Expected2 = 11;
    if (i < NumThreads) {
      if (check2[i] != Expected2) {
        printf("invalid: check2[%d] should be %d, is %d\n", i, Expected2, check2[i]);
      }
    } else if (check2[i] != 0) {
      printf("invalid: check2[%d] should be 0, is %d\n", i, check2[i]);
    }
  }

  return 0;
}
