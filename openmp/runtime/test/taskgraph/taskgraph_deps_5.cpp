// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env KMP_G_DEBUG=10 %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental, libomp_debug

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
          #pragma omp task depend(out: deps[0], deps[2])
          { }
          #pragma omp task depend(out: deps[1], deps[3])
          { }
          #pragma omp task depend(inoutset: deps[0], deps[1])
          { }
          #pragma omp task depend(inoutset: deps[0], deps[1])
          { }
          #pragma omp task depend(inoutset: deps[2], deps[3])
          { }
          #pragma omp task depend(inoutset: deps[2], deps[3])
          { }
          #pragma omp task depend(in: deps[0], deps[1])
          { }
          #pragma omp task depend(in: deps[2], deps[3])
          { }
        }
      }
    }
  }
  return 0;
}

// CHECK:      Processed taskgraph 0x[[#%x,GRAPHPTR:]] (graph_id 0):
// CHECK-NEXT: sequential {
// CHECK-NEXT:   parallel {
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   }
// CHECK-NEXT:   parallel {
// CHECK-NEXT:     sequential {
// CHECK-NEXT:       parallel {
// CHECK-NEXT:         node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:         node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       }
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     }
// CHECK-NEXT:     sequential {
// CHECK-NEXT:       parallel {
// CHECK-NEXT:         node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:         node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:       }
// CHECK-NEXT:       node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
