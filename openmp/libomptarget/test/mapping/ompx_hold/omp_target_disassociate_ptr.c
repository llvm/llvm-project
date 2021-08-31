// omp_target_disassociate_ptr should always fail if the hold reference count is
// non-zero, regardless of the dynamic reference count.  When the latter is
// finite, the implementation happens to choose to report the hold diagnostic.

// RUN: %libomptarget-compile-generic -fopenmp-extensions
// RUN: %not %libomptarget-run-generic 0   2>&1 | %fcheck-generic
// RUN: %not %libomptarget-run-generic 1   2>&1 | %fcheck-generic
// RUN: %not %libomptarget-run-generic inf 2>&1 | %fcheck-generic

// RUN: %libomptarget-compile-generic -fopenmp-extensions -DHOLD_MORE
// RUN: %not %libomptarget-run-generic 0   2>&1 | %fcheck-generic
// RUN: %not %libomptarget-run-generic 1   2>&1 | %fcheck-generic
// RUN: %not %libomptarget-run-generic inf 2>&1 | %fcheck-generic

#include <omp.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>

int main(int argc, char *argv[]) {
  // Parse command line.
  int DynRef;
  if (argc != 2) {
    fprintf(stderr, "bad arguments\n");
    return 1;
  }
  if (0 == strcmp(argv[1], "inf"))
    DynRef = INT_MAX;
  else
    DynRef = atoi(argv[1]);

  // Allocate and set dynamic reference count as specified.
  int DevNum = omp_get_default_device();
  int X;
  void *XDev = omp_target_alloc(sizeof X, DevNum);
  if (!XDev) {
    fprintf(stderr, "omp_target_alloc failed\n");
    return 1;
  }
  if (DynRef == INT_MAX) {
    if (omp_target_associate_ptr(&X, &XDev, sizeof X, 0, DevNum)) {
      fprintf(stderr, "omp_target_associate_ptr failed\n");
      return 1;
    }
  } else {
    for (int I = 0; I < DynRef; ++I) {
      #pragma omp target enter data map(alloc: X)
    }
  }

  // Disassociate while hold reference count > 0.
  int Status = 0;
  #pragma omp target data map(ompx_hold,alloc: X)
#if HOLD_MORE
  #pragma omp target data map(ompx_hold,alloc: X)
  #pragma omp target data map(ompx_hold,alloc: X)
#endif
  {
    //      CHECK: Libomptarget error: Trying to disassociate a pointer with a
    // CHECK-SAME: non-zero hold reference count
    // CHECK-NEXT: omp_target_disassociate_ptr failed
    if (omp_target_disassociate_ptr(&X, DevNum)) {
      fprintf(stderr, "omp_target_disassociate_ptr failed\n");
      Status = 1;
    }
  }
  return Status;
}
