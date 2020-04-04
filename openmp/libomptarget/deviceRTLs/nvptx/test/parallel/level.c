// RUN: %compile-run-and-check

#include <omp.h>
#include <stdio.h>

const int MaxThreads = 1024;
const int NumThreads = 64;

int main(int argc, char *argv[]) {
  int level = -1, activeLevel = -1;
  // The expected value is -1, initialize to different value.
  int ancestorTNumNeg = 1, teamSizeNeg = 1;
  int ancestorTNum0 = -1, teamSize0 = -1;
  // The expected value is -1, initialize to different value.
  int ancestorTNum1 = 1, teamSize1 = 1;
  int check1[MaxThreads];
  int check2[MaxThreads];
  int check3[MaxThreads];
  int check4[MaxThreads];
  for (int i = 0; i < MaxThreads; i++) {
    check1[i] = check2[i] = check3[i] = check4[i] = 0;
  }

  #pragma omp target map(level, activeLevel, ancestorTNumNeg, teamSizeNeg) \
                     map(ancestorTNum0, teamSize0, ancestorTNum1, teamSize1) \
                     map(check1[:], check2[:], check3[:], check4[:])
  {
    level = omp_get_level();
    activeLevel = omp_get_active_level();

    // Expected to return -1.
    ancestorTNumNeg = omp_get_ancestor_thread_num(-1);
    teamSizeNeg = omp_get_team_size(-1);

    // Expected to return 0 and 1.
    ancestorTNum0 = omp_get_ancestor_thread_num(0);
    teamSize0 = omp_get_team_size(0);

    // Expected to return -1 because the requested level is larger than
    // the nest level.
    ancestorTNum1 = omp_get_ancestor_thread_num(1);
    teamSize1 = omp_get_team_size(1);

    // Expecting active parallel region.
    #pragma omp parallel num_threads(NumThreads)
    {
      int id = omp_get_thread_num();
      // Multiply return value of omp_get_level by 5 to avoid that this test
      // passes if both API calls return wrong values.
      check1[id] += omp_get_level() * 5 + omp_get_active_level();

      // Expected to return 0 and 1.
      check2[id] += omp_get_ancestor_thread_num(0) + 5 * omp_get_team_size(0);
      // Expected to return the current thread num.
      check2[id] += (omp_get_ancestor_thread_num(1) - id);
      // Expected to return the current number of threads.
      check2[id] += 3 * omp_get_team_size(1);
      // Expected to return -1, see above.
      check2[id] += omp_get_ancestor_thread_num(2) + omp_get_team_size(2);

      // Expecting serialized parallel region.
      #pragma omp parallel
      {
        #pragma omp atomic
        check3[id] += omp_get_level() * 5 + omp_get_active_level();

        // Expected to return 0 and 1.
        int check4Inc = omp_get_ancestor_thread_num(0) + 5 * omp_get_team_size(0);
        // Expected to return the parent thread num.
        check4Inc += (omp_get_ancestor_thread_num(1) - id);
        // Expected to return the number of threads in the active parallel region.
        check4Inc += 3 * omp_get_team_size(1);
        // Expected to return 0 and 1.
        check4Inc += omp_get_ancestor_thread_num(2) + 3 * omp_get_team_size(2);
        // Expected to return -1, see above.
        check4Inc += omp_get_ancestor_thread_num(3) + omp_get_team_size(3);

        #pragma omp atomic
        check4[id] += check4Inc;
      }
    }
  }

  // CHECK: target: level = 0, activeLevel = 0
  printf("target: level = %d, activeLevel = %d\n", level, activeLevel);
  // CHECK: level = -1: ancestorTNum = -1, teamSize = -1
  printf("level = -1: ancestorTNum = %d, teamSize = %d\n", ancestorTNumNeg, teamSizeNeg);
  // CHECK: level = 0: ancestorTNum = 0, teamSize = 1
  printf("level = 0: ancestorTNum = %d, teamSize = %d\n", ancestorTNum0, teamSize0);
  // CHECK: level = 1: ancestorTNum = -1, teamSize = -1
  printf("level = 1: ancestorTNum = %d, teamSize = %d\n", ancestorTNum1, teamSize1);

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

    // 5 * 1 + 3 * 64 - 1 - 1 (see above)
    const int Expected2 = 195;
    if (i < NumThreads) {
      if (check2[i] != Expected2) {
        printf("invalid: check2[%d] should be %d, is %d\n", i, Expected2, check2[i]);
      }
    } else if (check2[i] != 0) {
      printf("invalid: check2[%d] should be 0, is %d\n", i, check2[i]);
    }

    // Check serialized parallel region:
    // omp_get_level() = 2, omp_get_active_level() = 1
    const int Expected3 = 11;
    if (i < NumThreads) {
      if (check3[i] != Expected3) {
        printf("invalid: check3[%d] should be %d, is %d\n", i, Expected3, check3[i]);
      }
    } else if (check3[i] != 0) {
      printf("invalid: check3[%d] should be 0, is %d\n", i, check3[i]);
    }

    // 5 * 1 + 3 * 64 + 3 * 1 - 1 - 1 (see above)
    const int Expected4 = 198;
    if (i < NumThreads) {
      if (check4[i] != Expected4) {
        printf("invalid: check4[%d] should be %d, is %d\n", i, Expected4, check4[i]);
      }
    } else if (check4[i] != 0) {
      printf("invalid: check4[%d] should be 0, is %d\n", i, check4[i]);
    }
  }

  // Check for paraller level in non-SPMD kernels.
  level = 0;
  #pragma omp target teams distribute num_teams(1) thread_limit(32) reduction(+:level)
  for (int i=0; i<5032; i+=32) {
    int ub = (i+32 > 5032) ? 5032 : i+32;
    #pragma omp parallel for schedule(dynamic)
    for (int j=i ; j < ub; j++) ;
    level += omp_get_level();
  }
  // CHECK: Integral level = 0.
  printf("Integral level = %d.\n", level);

  return 0;
}
