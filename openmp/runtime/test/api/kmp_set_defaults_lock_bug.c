// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"
/* The bug occurs if the lock table is reallocated after
   kmp_set_defaults() is called.  If the table is reallocated,
   then the lock will not point to a valid lock object after the
   kmp_set_defaults() call.*/
omp_lock_t lock;

int test_kmp_set_defaults_lock_bug()
{
  /* checks that omp_get_num_threads is equal to the number of
     threads */
  int nthreads_lib;
  int nthreads = 0;

  nthreads_lib = -1;

  #pragma omp parallel
  {
    omp_set_lock(&lock);
    nthreads++;
    omp_unset_lock(&lock);
    #pragma omp single
    {
      nthreads_lib = omp_get_num_threads ();
    }  /* end of single */
  } /* end of parallel */
  kmp_set_defaults("OMP_NUM_THREADS");
  #pragma omp parallel
  {
    omp_set_lock(&lock);
    nthreads++;
    omp_unset_lock(&lock);
  } /* end of parallel */

  return (nthreads == 2*nthreads_lib);
}

int main()
{
  int i;
  int num_failed=0;
  omp_init_lock(&lock);

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_kmp_set_defaults_lock_bug()) {
      num_failed++;
    }
  }
  omp_destroy_lock(&lock);
  return num_failed;
}
