// RUN: %libomp-cxx-compile -fopenmp-version=51
// RUN: %libomp-run | FileCheck %s --check-prefix OMP51

#include <stdio.h>
#include <omp.h>

void foo() {
#pragma omp parallel num_threads(10)
  { printf("\ntarget: foo(): parallel num_threads(10)"); }
}

int main(void) {

  int tl = 4;
  printf("\nmain: thread_limit = %d", omp_get_thread_limit());
  // OMP51: main: thread_limit = {{[0-9]+}}

#pragma omp target thread_limit(tl)
  {
    printf("\ntarget: thread_limit = %d", omp_get_thread_limit());
    int count = 0;
// OMP51: target: thread_limit = 4
// check whether thread_limit is honoured
#pragma omp parallel reduction(+:count)
    { count++; }
    printf("\ntarget: parallel: count = %d", count);
// OMP51: target: parallel: count = {{(1|2|3|4)$}}

// check whether num_threads is honoured
#pragma omp parallel num_threads(2)
    { printf("\ntarget: parallel num_threads(2)"); }
// OMP51: target: parallel num_threads(2)
// OMP51: target: parallel num_threads(2)
// OMP51-NOT: target: parallel num_threads(2)

// check whether thread_limit is honoured when there is a conflicting
// num_threads
#pragma omp parallel num_threads(10)
    { printf("\ntarget: parallel num_threads(10)"); }
    // OMP51: target: parallel num_threads(10)
    // OMP51: target: parallel num_threads(10)
    // OMP51: target: parallel num_threads(10)
    // OMP51: target: parallel num_threads(10)
    // OMP51-NOT: target: parallel num_threads(10)

    // check whether threads are limited across functions
    foo();
    // OMP51: target: foo(): parallel num_threads(10)
    // OMP51: target: foo(): parallel num_threads(10)
    // OMP51: target: foo(): parallel num_threads(10)
    // OMP51: target: foo(): parallel num_threads(10)
    // OMP51-NOT: target: foo(): parallel num_threads(10)

    // check if user can set num_threads at runtime
    omp_set_num_threads(2);
#pragma omp parallel
    { printf("\ntarget: parallel with omp_set_num_thread(2)"); }
    // OMP51: target: parallel with omp_set_num_thread(2)
    // OMP51: target: parallel with omp_set_num_thread(2)
    // OMP51-NOT: target: parallel with omp_set_num_thread(2)

    // make sure thread_limit is unaffected by omp_set_num_threads
    printf("\ntarget: thread_limit = %d", omp_get_thread_limit());
    // OMP51: target: thread_limit = 4
  }

// checking consecutive target regions with different thread_limits
#pragma omp target thread_limit(3)
  {
    printf("\nsecond target: thread_limit = %d", omp_get_thread_limit());
    int count = 0;
// OMP51: second target: thread_limit = 3
#pragma omp parallel reduction(+:count)
    { count++; }
    printf("\nsecond target: parallel: count = %d", count);
    // OMP51: second target: parallel: count = {{(1|2|3)$}}
  }

  // confirm that thread_limit's effects are limited to target region
  printf("\nmain: thread_limit = %d", omp_get_thread_limit());
  // OMP51: main: thread_limit = {{[0-9]+}}
#pragma omp parallel num_threads(10)
  { printf("\nmain: parallel num_threads(10)"); }
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51: main: parallel num_threads(10)
  // OMP51-NOT: main: parallel num_threads(10)

// check combined target directives which support thread_limit
// target parallel
#pragma omp target parallel thread_limit(2)
  printf("\ntarget parallel thread_limit(2)");
  // OMP51: target parallel thread_limit(2)
  // OMP51: target parallel thread_limit(2)
  // OMP51-NOT: target parallel thread_limit(2)

#pragma omp target parallel num_threads(2) thread_limit(3)
  printf("\ntarget parallel num_threads(2) thread_limit(3)");
  // OMP51: target parallel num_threads(2) thread_limit(3)
  // OMP51: target parallel num_threads(2) thread_limit(3)
  // OMP51-NOT: target parallel num_threads(2) thread_limit(3)

#pragma omp target parallel num_threads(3) thread_limit(2)
  printf("\ntarget parallel num_threads(3) thread_limit(2)");
  // OMP51: target parallel num_threads(3) thread_limit(2)
  // OMP51: target parallel num_threads(3) thread_limit(2)
  // OMP51-NOT: target parallel num_threads(3) thread_limit(2)

// target parallel for
#pragma omp target parallel for thread_limit(2)
  for (int i = 0; i < 5; ++i)
    printf("\ntarget parallel for thread_limit(2) : thread num = %d",
           omp_get_thread_num());
    // OMP51: target parallel for thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for thread_limit(2) : thread num = {{0|1}}
    // OMP51-NOT: target parallel for thread_limit(3) : thread num = {{0|1}}

// target parallel for simd
#pragma omp target parallel for simd thread_limit(2)
  for (int i = 0; i < 5; ++i)
    printf("\ntarget parallel for simd thread_limit(2) : thread num = %d",
           omp_get_thread_num());
    // OMP51: target parallel for simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target parallel for simd thread_limit(2) : thread num = {{0|1}}
    // OMP51-NOT: target parallel for simd thread_limit(2) : thread num =
    // {{0|1}}

// target simd
#pragma omp target simd thread_limit(2)
  for (int i = 0; i < 5; ++i)
    printf("\ntarget simd thread_limit(2) : thread num = %d",
           omp_get_thread_num());
    // OMP51: target simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target simd thread_limit(2) : thread num = {{0|1}}
    // OMP51: target simd thread_limit(2) : thread num = {{0|1}}
    // OMP51-NOT: target simd thread_limit(2) : thread num = {{0|1}}

// target parallel loop
#pragma omp target parallel loop thread_limit(2)
  for (int i = 0; i < 5; ++i)
    printf("\ntarget parallel loop thread_limit(2) : thread num = %d",
           omp_get_thread_num());
  // # OMP51: target parallel loop thread_limit(2) : thread num = {{0|1}}
  // # OMP51: target parallel loop thread_limit(2) : thread num = {{0|1}}
  // # OMP51: target parallel loop thread_limit(2) : thread num = {{0|1}}
  // # OMP51: target parallel loop thread_limit(2) : thread num = {{0|1}}
  // # OMP51: target parallel loop thread_limit(2) : thread num = {{0|1}}
  // # OMP51-NOT: target parallel loop thread_limit(2) : thread num = {{0|1}}
  return 0;
}
