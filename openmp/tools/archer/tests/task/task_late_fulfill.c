// REQUIRES: garbage
// RUN: %libarcher-compile -fopenmp-version=50 && env OMP_NUM_THREADS='3' \
// RUN:   %libarcher-run-race | FileCheck %s

// Checked gcc 10.1 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10
// gcc 11 introduced detach clause, but gomp interface in libomp has no support
// XFAIL: gcc-11, gcc-12
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc
// REQUIRES: tsan

#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main() {
#pragma omp parallel
#pragma omp master
  {
    omp_event_handle_t event;
    int a = 0, b = 0;
    omp_event_handle_t *f_event;
#pragma omp task detach(event) depend(out : f_event) shared(f_event)
    {
      printf("%i: task 1\n", omp_get_thread_num());
      f_event = &event;
    }
    usleep(10000);
#pragma omp task depend(in : f_event) shared(f_event, a, b)
    {
      printf("%i: task 2, %p, %i, %i\n", omp_get_thread_num(), f_event, a, b);
      f_event = &event;
    }
    usleep(10000);
    a++;
    printf("%i: calling omp_fulfill_event\n", omp_get_thread_num());
    omp_fulfill_event(event);
//#pragma omp task if (0) depend(in : f_event)
//    {}
    b++;
    usleep(10000);
#pragma omp taskwait
  }
  return 0;
}

// no race for a++ in line 32:
// CHECK-NOT: #0 {{.*}}task_late_fulfill.c:37

// CHECK: WARNING: ThreadSanitizer: data race
// CHECK-NEXT:   {{(Write|Read)}} of size 4
// CHECK-NEXT: #0 {{.*}}task_late_fulfill.c:33
// CHECK:   Previous write of size 4
// CHECK-NEXT: #0 {{.*}}task_late_fulfill.c:42
