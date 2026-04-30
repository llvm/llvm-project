// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <cstdio>

void foo() {
  fprintf(stderr, "called function foo\n");
#pragma omp taskloop replayable num_tasks(4)
  {
    for (int i = 0; i < 4; i++)
      fprintf(stderr, "taskloop iter %d outside lexical taskgraph\n", i);
  }
}

int main()
{
  int arr[100];

  int res = 0;
  for (int i = 0; i < 100; i++) {
    arr[i] = i;
    res += i;
  }
  fprintf(stderr, "base result: %d\n", res);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 10; i++)
      {
        int res = 0;
        #pragma omp taskgraph
        {
          #pragma omp taskloop reduction(+: res) num_tasks(10)
          {
            for (int j = 0; j < 100; j++) {
              res += arr[j];
            }
          }
          foo();
        }
        fprintf(stderr, "reduction result: %d\n", res);
      }
    }
  }
  return 0;
}

// CHECK: base result: 4950
// CHECK-NEXT: called function foo
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: taskloop iter 0 outside lexical taskgraph
// CHECK-DAG: taskloop iter 1 outside lexical taskgraph
// CHECK-DAG: taskloop iter 2 outside lexical taskgraph
// CHECK-DAG: taskloop iter 3 outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
