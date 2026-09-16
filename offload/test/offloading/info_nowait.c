// Verify LIBOMPTARGET_INFO reporting of data-transfer synchronization.
//
// RUN: %libomptarget-compile-generic -gline-tables-only -fopenmp-extensions
// RUN: env LIBOMPTARGET_INFO=32 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic -allow-empty -check-prefixes=INFO

// FIXME: Fails due to optimized debugging in 'ptxas'.
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO

#include <stdio.h>

int main(void) {
  int x = 0, y = 0;

  // Blocking region: synchronization is blocking, so the wait is announced.
#pragma omp target map(tofrom : x)
  x = 1;

  // Nowait kernel inside an active parallel/single: it has a task team and
  // therefore synchronizes non-blockingly, so no wait must be announced even
  // though its data transfers are still reported.
#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp target map(tofrom : x) nowait
    x = 2;
#pragma omp taskwait
  }

  // Nowait target-data constructs reach the synchronization notice through a
  // different runtime entry point; they are non-blocking here too and likewise
  // must not announce a wait.
#pragma omp parallel num_threads(2)
#pragma omp single
  {
#pragma omp target enter data map(to : y) nowait
#pragma omp taskwait
#pragma omp target exit data map(from : y) nowait
#pragma omp taskwait
  }

  printf("x = %d, y = %d\n", x, y);
  return x != 2;
}

// clang-format off
// INFO: info: Waiting for asynchronous operations to complete at info_nowait.c:{{[0-9]+}}:{{[0-9]+}}
// INFO: info: Copying data from host to device,{{.*}}at info_nowait.c:{{[0-9]+}}:{{[0-9]+}}
// INFO-NOT: Waiting for asynchronous operations to complete
// clang-format on
