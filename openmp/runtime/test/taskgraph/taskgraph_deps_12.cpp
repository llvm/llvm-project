// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env KMP_G_DEBUG=10 %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental, libomp_debug

int main()
{
  int deps[2];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 2; i++)
      {
        #pragma omp taskgraph
        {
          #pragma omp task depend(mutexinoutset: deps[0])
          { }
          #pragma omp task depend(mutexinoutset: deps[1])
          { }
          #pragma omp task depend(mutexinoutset: deps[0])
          { }
          #pragma omp task depend(mutexinoutset: deps[1])
          { }
          #pragma omp task depend(mutexinoutset: deps[0])
          { }
          #pragma omp task depend(mutexinoutset: deps[1])
          { }
          #pragma omp task depend(mutexinoutset: deps[0])
          { }
          #pragma omp task depend(mutexinoutset: deps[1])
          { }
        }
      }
    }
  }
  return 0;
}

// CHECK:      Processed taskgraph 0x[[#%x,GRAPHPTR:]] (graph_id 0):
// CHECK-NEXT: parallel {
// CHECK-NEXT:   exclusive {
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   }
// CHECK-NEXT:   exclusive {
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
