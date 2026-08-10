// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "omp_my_sleep.h"

int a = 0;

void task1() {
  my_sleep(0.5);
  a = 10;
}

void task2() {
  a++;
}

int main(int argc, char** argv)
{
  #pragma omp parallel shared(argc) num_threads(2)
  {
    #pragma omp single
    {
      #pragma omp task depend(out: a)
      task1();

      #pragma omp task if(0) depend(inout: a)
      task2();
    }
  }
  if (a != 11) {
    fprintf(stderr, "fail: expected 11, but a is %d\n", a);
    exit(1);
  } else {
    printf("pass\n");
  }
  return 0;
}
