// RUN: %libomptarget-compile-generic -DCLAUSE=to
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -DCLAUSE=from
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// UNSUPPORTED: apu

// A data-motion clause treats an array section as its elements, so a
// zero-length one has nothing to update and nothing to check for presence.
// Unlike map(present: arr[0:0]) (zero_length_array_section.c), this is not an
// error even if arr is not mapped.

#include <assert.h>
#include <stdio.h>

int main(void) {
  int arr[5] = {-1, -1, -1, -1, -1};

  // CHECK-NOT: omptarget
#pragma omp target update CLAUSE(present : arr[0 : 0])

  // Empty updates also preserve both the host and device values of a mapping.
#pragma omp target data map(to : arr)
  {
    for (int i = 0; i < 5; ++i)
      arr[i] = 7;
#pragma omp target update CLAUSE(present : arr[0 : 0])
    for (int i = 0; i < 5; ++i)
      assert(arr[i] == 7);
#pragma omp target update from(arr)
    for (int i = 0; i < 5; ++i)
      assert(arr[i] == -1);
  }

  // CHECK: done
  fprintf(stderr, "done\n");
  return 0;
}
