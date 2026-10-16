// --------------------------------------------------
// Check extends before
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DEXTENDS=BEFORE
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// --------------------------------------------------
// Check extends after
// --------------------------------------------------

// RUN: %libomptarget-compile-generic \
// RUN:   -DEXTENDS=AFTER
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// END.

#include <stdio.h>

#define BEFORE 0
#define AFTER 1

#define SIZE 100

#if EXTENDS == BEFORE
#define SMALL_BEG (SIZE - 2)
#define SMALL_END SIZE
#define LARGE_BEG 0
#define LARGE_END SIZE
#elif EXTENDS == AFTER
#define SMALL_BEG 0
#define SMALL_END 2
#define LARGE_BEG 0
#define LARGE_END SIZE
#else
#error EXTENDS undefined
#endif

#define SMALL                                                                  \
  SMALL_BEG:                                                                   \
  (SMALL_END - SMALL_BEG)
#define LARGE                                                                  \
  LARGE_BEG:                                                                   \
  (LARGE_END - LARGE_BEG)

void check_not_present() {
  int arr[SIZE];

  for (int i = 0; i < SIZE; ++i)
    arr[i] = 99;

  // CHECK-LABEL: checking not present
  fprintf(stderr, "checking not present\n");

  // arr[LARGE] isn't (fully) present at the end of the target data region, so
  // the device-to-host transfer should not be performed, or it might fail.
#pragma omp target data map(tofrom : arr[LARGE])
  {
#pragma omp target exit data map(delete : arr[LARGE])
#pragma omp target enter data map(alloc : arr[SMALL])
#pragma omp target map(alloc : arr[SMALL])
    for (int i = SMALL_BEG; i < SMALL_END; ++i)
      arr[i] = 88;
  }

  // CHECK-NOT: omptarget
  // CHECK-NOT: error
  for (int i = 0; i < SIZE; ++i) {
    if (arr[i] != 99)
      fprintf(stderr, "error: arr[%d]=%d\n", i, arr[i]);
  }
}

void check_is_present() {
  int arr[SIZE];

  for (int i = 0; i < SIZE; ++i)
    arr[i] = 99;

  // CHECK-LABEL: checking is present
  fprintf(stderr, "checking is present\n");

  // arr[SMALL] is (fully) present at the end of the target data region, and the
  // device-to-host transfer should be performed only for it even though more
  // of the array is then present.
#pragma omp target data map(tofrom : arr[SMALL])
  {
#pragma omp target exit data map(delete : arr[SMALL])
#pragma omp target enter data map(alloc : arr[LARGE])
#pragma omp target map(alloc : arr[LARGE])
    for (int i = LARGE_BEG; i < LARGE_END; ++i)
      arr[i] = 88;
  }

  // CHECK-NOT: omptarget
  // CHECK-NOT: error
  for (int i = 0; i < SIZE; ++i) {
    if (SMALL_BEG <= i && i < SMALL_END) {
      if (arr[i] != 88)
        fprintf(stderr, "error: arr[%d]=%d\n", i, arr[i]);
    } else if (arr[i] != 99) {
      fprintf(stderr, "error: arr[%d]=%d\n", i, arr[i]);
    }
  }
}

int main() {
  check_not_present();
  check_is_present();
  return 0;
}
