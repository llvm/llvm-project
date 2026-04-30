// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <cstdio>

void foo() {
#pragma omp task replayable(1)
  {
    fprintf(stderr, "task outside lexical taskgraph\n");
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
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
// CHECK-DAG: task outside lexical taskgraph
// CHECK-DAG: reduction result: 4950
