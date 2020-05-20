// RUN: %libomp-compile -fopenmp-version=50 && env OMP_NUM_THREADS='3' %libomp-run
// RUN: %libomp-compile -fopenmp-version=50 && env OMP_NUM_THREADS='1' %libomp-run

// Checked gcc 9.2 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc

#include <omp.h>

int main()
{
  #pragma omp parallel
  #pragma omp master
  {
    omp_event_handle_t event;
    #pragma omp task detach(event)
    {
      omp_fulfill_event(event);
    }
    #pragma omp taskwait
  }
  return 0;
}
