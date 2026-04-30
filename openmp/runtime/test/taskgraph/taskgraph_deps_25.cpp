// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <cstdio>

int global_dep;

void foo() {
  fprintf(stderr, "called function foo\n");
#pragma omp task replayable(1) depend(in: global_dep)
  {
    fprintf(stderr, "out-of-line task created from within taskloop\n");
  }
}

int main()
{
  int arr[100];

  int res = 0;
  for (int i = 0; i < 4; i++) {
    arr[i] = i;
    res += i;
  }
  fprintf(stderr, "base result: %d\n", res);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 4; i++)
      {
        int res = 0;
        #pragma omp taskgraph
        {
          #pragma omp taskloop reduction(+: res) num_tasks(4)
          {
            for (int j = 0; j < 4; j++) {
              res += arr[j];
              foo();
            }
          }
        }
        fprintf(stderr, "reduction result: %d\n", res);
      }
    }
  }
  return 0;
}

// CHECK: base result: 6
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: reduction result: 6
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: reduction result: 6
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: reduction result: 6
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: called function foo
// CHECK-DAG: out-of-line task created from within taskloop
// CHECK-DAG: reduction result: 6
