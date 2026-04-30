// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env KMP_G_DEBUG=10 %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental, libomp_debug

#include <cassert>

int global_dep;

void foo() {
#pragma omp taskwait replayable(1) depend(in: global_dep)
}

int main()
{
  int arr[100];

  int res = 0;
  for (int i = 0; i < 100; i++) {
    arr[i] = i;
    res += i;
  }

  assert(res == 4950);

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
          #pragma omp task depend(out: global_dep)
          { }
          foo();
        }
        assert(res == 4950);
      }
    }
  }
  return 0;
}

// CHECK:      Processed taskgraph 0x[[#%x,GRAPHPTR:]] (graph_id 0):
// CHECK-NEXT: sequential {
// CHECK-NEXT:   sequential {
// CHECK-NEXT:     parallel {
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     }
// CHECK-NEXT:     wait: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   }
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   wait: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: }
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
