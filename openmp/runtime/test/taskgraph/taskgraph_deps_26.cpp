// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

#include <cstdio>

int main()
{
  int arr[100];
  int arr2[100];

  int res = 0, res2 = 0;
  for (int i = 0; i < 10; i++) {
    arr[i] = i;
    arr2[i] = 3 + i * 2;
    res += i;
    res2 += 3 + i * 2;
  }
  fprintf(stderr, "base results: %d, %d\n", res, res2);

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 10; i++)
      {
        int res = 0, res2 = 0;
        #pragma omp taskgraph
        {
          #pragma omp taskloop reduction(+: res) num_tasks(10)
          {
            for (int j = 0; j < 10; j++) {
              res += arr[j];
            }
          }
          #pragma omp taskloop reduction(+: res2) num_tasks(10)
          {
            for (int j = 0; j < 10; j++) {
              res2 += arr2[j];
            }
          }
        }
        fprintf(stderr, "reduction results: %d, %d\n", res, res2);
      }
    }
  }
  return 0;
}

// CHECK: base results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
// CHECK-NEXT: reduction results: 45, 120
