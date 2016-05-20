// RUN: %libomp-compile-and-run
/*
 * Test for dynamic scheduling with chunk size
 * Method: caculate how many times the iteration space is dispatched
 *     and judge if each dispatch has the requested chunk size
 *     unless it is the last one.
 * It is possible for two adjacent chunks are assigned to the same thread
 * Modified by Chunhua Liao
 */
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "omp_testsuite.h"

#define CFDMAX_SIZE 100
const int chunk_size = 7;

int test_omp_for_schedule_dynamic()
{
  int tid;
  int *tids;
  int i;
  int tidsArray[CFDMAX_SIZE];
  int count = 0;
  int tmp_count = 0; /*dispatch times*/
  int *tmp;  /*store chunk size for each dispatch*/
  int result = 0;

  tids = tidsArray;

  #pragma omp parallel private(tid) shared(tids)
  {        /* begin of parallel */
    int tid;
    tid = omp_get_thread_num ();
    #pragma omp for schedule(dynamic,chunk_size)
    for (i = 0; i < CFDMAX_SIZE; i++) {
      tids[i] = tid;
    }
  }

  for (i = 0; i < CFDMAX_SIZE - 1; ++i) {
    if (tids[i] != tids[i + 1]) {
      count++;
    }
  }

  tmp = (int *) malloc (sizeof (int) * (count + 1));
  tmp[0] = 1;

  for (i = 0; i < CFDMAX_SIZE - 1; ++i) {
    if (tmp_count > count) {
      printf ("--------------------\nTestinternal Error: List too small!!!\n--------------------\n");  /* Error handling */
      break;
    }
    if (tids[i] != tids[i + 1]) {
      tmp_count++;
      tmp[tmp_count] = 1;
    } else {
      tmp[tmp_count]++;
    }
  }
  /* is dynamic statement working? */
  for (i = 0; i < count; i++) {
    if ((tmp[i]%chunk_size)!=0) {
      /* it is possible for 2 adjacent chunks assigned to a same thread */
      result++;
      fprintf(stderr,"The intermediate dispatch has wrong chunksize.\n");
      /* result += ((tmp[i] / chunk_size) - 1); */
    }
  }
  if ((tmp[count]%chunk_size)!=(CFDMAX_SIZE%chunk_size)) {
    result++;
    fprintf(stderr,"the last dispatch has wrong chunksize.\n");
  }
  /* for (int i=0;i<count+1;++i) printf("%d\t:=\t%d\n",i+1,tmp[i]); */
  return (result==0);
}
int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_schedule_dynamic()) {
      num_failed++;
    }
  }
  return num_failed;
}
