#include <omp.h>
#include <malloc.h>
#include <stdio.h>
#include <memory.h>

#define LOOP_IV_TYPE0 LOOP_TYPES
#define LOOP_TYPE0 LOOP_TYPES
#define LOOP_STYPE0 LOOP_TYPES

#define LOOP_IV_TYPE1 LOOP_TYPES
#define LOOP_TYPE1 LOOP_TYPES
#define LOOP_STYPE1 LOOP_TYPES

#define LOOP_IV_TYPE2 LOOP_TYPES
#define LOOP_TYPE2 LOOP_TYPES
#define LOOP_STYPE2 LOOP_TYPES

#define MAX_THREADS 256

#if defined VERBOSE
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif

LOOP_TYPE0 iLB, iUB;
LOOP_TYPE1 jA0, jB0;
LOOP_TYPE2 kA0, kB0;

LOOP_STYPE0 iStep;
LOOP_STYPE1 jA1, jB1, jStep;
LOOP_STYPE2 kA1, kB1, kStep;

// We can check <=, <, >=, > (!= has different pattern)
// Additional definition of LOOP_LEi, LOOP_LTi, etc. is helpful to build calls
// of the test from main

#if defined LOOP_LE0
#define COMPARE0 <=
#elif defined LOOP_LT0
#define COMPARE0 <
#elif defined LOOP_GE0
#define COMPARE0 >=
#elif defined LOOP_GT0
#define COMPARE0 >
#endif

#if defined LOOP_LE1
#define COMPARE1 <=
#elif defined LOOP_LT1
#define COMPARE1 <
#elif defined LOOP_GE1
#define COMPARE1 >=
#elif defined LOOP_GT1
#define COMPARE1 >
#endif

#if defined LOOP_LE2
#define COMPARE2 <=
#elif defined LOOP_LT2
#define COMPARE2 <
#elif defined LOOP_GE2
#define COMPARE2 >=
#elif defined LOOP_GT2
#define COMPARE2 >
#endif

typedef struct {
  LOOP_IV_TYPE0 i;
  LOOP_IV_TYPE1 j;
  LOOP_IV_TYPE2 k;
} spaceType;

spaceType *AllocSpace(unsigned size) {

  spaceType *p = (spaceType *)malloc(size * sizeof(spaceType));
  memset(p, 0, size * sizeof(spaceType));
  return p;
}

void FreeSpace(spaceType *space) { free(space); }

// record an iteration
void Set(spaceType *space, unsigned count, unsigned trueCount, LOOP_IV_TYPE0 i,
         LOOP_IV_TYPE1 j, LOOP_IV_TYPE0 k) {
  if (count > trueCount) {
    // number of iterations exceeded
    // will be reported with checks
    return;
  }
  space[count - 1].i = i;
  space[count - 1].j = j;
  space[count - 1].k = k;
}
int test() {
  int pass = 1;
  LOOP_IV_TYPE0 i;
  LOOP_IV_TYPE1 j;
  LOOP_IV_TYPE2 k;

  spaceType *openmpSpace;
  spaceType *scalarSpace;

  unsigned trueCount = 0;
  unsigned openmpCount = 0;
  unsigned scalarCount = 0;
  unsigned uselessThreadsOpenMP = 0;
  unsigned usefulThreadsOpenMP = 0;

  // Use half of the available threads/logical processors.
  unsigned num_threads = omp_get_max_threads() / 2;

  // Make sure num_threads is not 0 after the division in case
  // omp_get_max_threads() returns 1.
  if (num_threads == 0)
    num_threads = 1;

  if (num_threads > MAX_THREADS)
    num_threads = MAX_THREADS;

  unsigned long *chunkSizesOpenmp =
      (unsigned long *)malloc(sizeof(unsigned long) * num_threads);
  memset(chunkSizesOpenmp, 0, sizeof(unsigned long) * num_threads);

  // count iterations and allocate space
  LOOP { ++trueCount; }

  openmpSpace = AllocSpace(trueCount);
  scalarSpace = AllocSpace(trueCount);

  // fill the scalar (compare) space
  LOOP {
    ++scalarCount;
    Set(scalarSpace, scalarCount, trueCount, i, j, k);
  }

  // test run body:
  // perform and record OpenMP iterations and thread use
#pragma omp parallel num_threads(num_threads)
  {
    unsigned gtid = omp_get_thread_num();
#pragma omp for collapse(3) private(i, j, k)
    LOOP {
      unsigned count;
#pragma omp atomic update
      ++chunkSizesOpenmp[gtid];
#pragma omp atomic capture
      count = ++openmpCount;
      Set(openmpSpace, count, trueCount, i, j, k);
    }
  }

  // check for the right number of iterations processed
  // (only need to check for less, greater is checked when recording)
  if (openmpCount < trueCount) {
    PRINTF("OpenMP FAILURE: Openmp processed fewer iterations: %d vs %d\n",
           openmpCount, trueCount);
    pass = 0;
  } else if (openmpCount > trueCount) {
    PRINTF("OpenMP FAILURE: Openmp processed more iterations: %d vs %d\n",
           openmpCount, trueCount);
    pass = 0;
  }

  // check openMP for iteration correctnes against scalar
  for (unsigned i = 0; i < trueCount; i++) {
    unsigned j;
    for (j = 0; j < openmpCount; j++) {
      if ((scalarSpace[i].i == openmpSpace[j].i) &&
          (scalarSpace[i].j == openmpSpace[j].j) &&
          (scalarSpace[i].k == openmpSpace[j].k)) {
        break;
      }
    }
    if (j == openmpCount) {
      PRINTF("OpenMP FAILURE: (%d %d %d) not processed\n", scalarSpace[i].i,
             scalarSpace[i].j, scalarSpace[i].k);
      pass = 0;
    }
  }

  // check for efficient thread use
  for (unsigned i = 0; i < num_threads; ++i) {
    if (chunkSizesOpenmp[i] == 0) {
      ++uselessThreadsOpenMP;
    }
  }

  // a check to see if at least more than one thread was used (weakish)
  if ((uselessThreadsOpenMP == num_threads - 1) && (trueCount > 1)) {
    PRINTF("OpenMP FAILURE: threads are not used\n");
    pass = 0;
  }

#if 0
    // a check to see if the load was spread more or less evenly so that
    // when there was more work than threads each one got at least something 
    // (stronger, but may currently fail for a general collapse case)
    if ((trueCount >= num_threads) && (uselessThreadsOpenMP > 0)) {
       PRINTF("OpenMP FAILURE: %d threads not used with %d iterations\n", 
           uselessThreadsOpenMP, openmpCount);
       pass = 0;
    }
#endif

  // clean up space
  FreeSpace(openmpSpace);
  FreeSpace(scalarSpace);
  free(chunkSizesOpenmp);
  return pass;
}
