// RUN: %libomp-compile-and-run

/* Test for guided scheduling
 * Ensure threads get chunks interleavely first
 * Then judge the chunk sizes are decreasing to a stable value
 * Modified by Chunhua Liao
 * For example, 100 iteration on 2 threads, chunksize 7
 * one line for each dispatch, 0/1 means thread id
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  24
 * 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1        18
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0            14
 * 1 1 1 1 1 1 1 1 1 1                10
 * 0 0 0 0 0 0 0 0                   8
 * 1 1 1 1 1 1 1                   7
 * 0 0 0 0 0 0 0                   7
 * 1 1 1 1 1 1 1                   7
 * 0 0 0 0 0                     5
*/
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

#define CFSMAX_SIZE 1000
#define MAX_TIME  0.005

#ifdef SLEEPTIME
#undef SLEEPTIME
#define SLEEPTIME 0.0001
#endif

int test_omp_for_schedule_guided()
{
  int * tids;
  int * chunksizes;
  int notout;
  int maxiter;
  int threads;
  int i;
  int result;

  tids = (int *) malloc (sizeof (int) * (CFSMAX_SIZE + 1));
  maxiter = 0;
  result = 1;
  notout = 1;

  /* Testing if enough threads are available for this check. */
  #pragma omp parallel
  {
    #pragma omp single
    {
      threads = omp_get_num_threads();
    }
  }

  /* ensure there are at least two threads */
  if (threads < 2) {
    omp_set_num_threads(2);
    threads = 2;
  }

  /* Now the real parallel work:
   * Each thread will start immediately with the first chunk.
   */
  #pragma omp parallel shared(tids,maxiter)
  {  /* begin of parallel */
    double count;
    int tid;
    int j;

    tid = omp_get_thread_num ();

    #pragma omp for nowait schedule(guided)
    for(j = 0; j < CFSMAX_SIZE; ++j) {
      count = 0.;
      #pragma omp flush(maxiter)
      if (j > maxiter) {
        #pragma omp critical
        {
          maxiter = j;
        }
      }
      /*printf ("thread %d sleeping\n", tid);*/
      #pragma omp flush(maxiter,notout)
      while (notout && (count < MAX_TIME) && (maxiter == j)) {
        #pragma omp flush(maxiter,notout)
        my_sleep (SLEEPTIME);
        count += SLEEPTIME;
#ifdef VERBOSE
        printf(".");
#endif
      }
#ifdef VERBOSE
      if (count > 0.) printf(" waited %lf s\n", count);
#endif
      /*printf ("thread %d awake\n", tid);*/
      tids[j] = tid;
#ifdef VERBOSE
      printf("%d finished by %d\n",j,tid);
#endif
    } /* end of for */
    notout = 0;
    #pragma omp flush(maxiter,notout)
  } /* end of parallel */

  /*******************************************************
   * evaluation of the values              *
   *******************************************************/
  {
    int determined_chunksize = 1;
    int last_threadnr = tids[0];
    int global_chunknr = 0;
    int openwork = CFSMAX_SIZE;
    int expected_chunk_size;
    int* local_chunknr = (int*)malloc(threads * sizeof(int));
    double c = 1;

    for (i = 0; i < threads; i++)
      local_chunknr[i] = 0;

    tids[CFSMAX_SIZE] = -1;

    /*
     * determine the number of global chunks
     */
    // fprintf(stderr,"# global_chunknr thread local_chunknr chunksize\n");
    for(i = 1; i <= CFSMAX_SIZE; ++i) {
      if (last_threadnr==tids[i]) {
        determined_chunksize++;
      } else {
        /* fprintf(stderr, "%d\t%d\t%d\t%d\n", global_chunknr,
           last_threadnr, local_chunknr[last_threadnr], m); */
        global_chunknr++;
        local_chunknr[last_threadnr]++;
        last_threadnr = tids[i];
        determined_chunksize = 1;
      }
    }
    /* now allocate the memory for saving the sizes of the global chunks */
    chunksizes = (int*)malloc(global_chunknr * sizeof(int));

    /*
    * Evaluate the sizes of the global chunks
    */
    global_chunknr = 0;
    determined_chunksize = 1;
    last_threadnr = tids[0];
    for (i = 1; i <= CFSMAX_SIZE; ++i) {
      /* If the threadnumber was the same as before increase the
       * detected chunksize for this chunk otherwise set the detected
       * chunksize again to one and save the number of the next
       * thread in last_threadnr.
       */
      if (last_threadnr == tids[i]) {
        determined_chunksize++;
      } else {
        chunksizes[global_chunknr] = determined_chunksize;
        global_chunknr++;
        local_chunknr[last_threadnr]++;
        last_threadnr = tids[i];
        determined_chunksize = 1;
      }
    }

#ifdef VERBOSE
    fprintf(stderr, "found\texpected\tconstant\n");
#endif

    /* identify the constant c for the exponential
       decrease of the chunksize */
    expected_chunk_size = openwork / threads;
    c = (double) chunksizes[0] / expected_chunk_size;

    for (i = 0; i < global_chunknr; i++) {
      /* calculate the new expected chunksize */
      if (expected_chunk_size > 1)
        expected_chunk_size = c * openwork / threads;
#ifdef VERBOSE
      fprintf(stderr, "%8d\t%8d\t%lf\n", chunksizes[i],
        expected_chunk_size, c * chunksizes[i]/expected_chunk_size);
#endif
      /* check if chunksize is inside the rounding errors */
      if (abs (chunksizes[i] - expected_chunk_size) >= 2) {
        result = 0;
#ifndef VERBOSE
        fprintf(stderr, "Chunksize differed from expected "
          "value: %d instead of %d\n", chunksizes[i],
          expected_chunk_size);
        return 0;
#endif
      } /* end if */

#ifndef VERBOSE
      if (expected_chunk_size - chunksizes[i] < 0)
        fprintf(stderr, "Chunksize did not decrease: %d"
          " instead of %d\n", chunksizes[i],expected_chunk_size);
#endif

      /* calculating the remaining amount of work */
      openwork -= chunksizes[i];
    }
  }
  return result;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_schedule_guided()) {
      num_failed++;
    }
  }
  return num_failed;
}
