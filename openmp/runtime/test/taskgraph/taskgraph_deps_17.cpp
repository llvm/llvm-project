// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <cstdio>

int main()
{
  int deps[4];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 2; i++)
      {
        #pragma omp taskgraph
        {
          #pragma omp task depend(out: deps[0], deps[1])
          {
            fprintf(stderr, "task 0\n");
          }
          #pragma omp task depend(out: deps[2], deps[3])
          {
            fprintf(stderr, "task 1\n");
          }
          #pragma omp task depend(inout: deps[0])
          {
            fprintf(stderr, "task 2\n");
          }
          #pragma omp task depend(inout: deps[1])
          {
            fprintf(stderr, "task 3\n");
          }
          #pragma omp task depend(inout: deps[2])
          {
            fprintf(stderr, "task 4\n");
          }
          #pragma omp task depend(inout: deps[3])
          {
            fprintf(stderr, "task 5\n");
          }
          #pragma omp task depend(in: deps[0], deps[1], deps[2], deps[3])
          {
            fprintf(stderr, "task 6\n");
          }
        }
      }
    }
  }
  return 0;
}

// CHECK-DAG: task 0
// CHECK-DAG: task 2
// CHECK-DAG: task 3
// CHECK-DAG: task 1
// CHECK-DAG: task 4
// CHECK-DAG: task 5
// CHECK: task 6

// CHECK-DAG: task 0
// CHECK-DAG: task 2
// CHECK-DAG: task 3
// CHECK-DAG: task 1
// CHECK-DAG: task 4
// CHECK-DAG: task 5
// CHECK: task 6
