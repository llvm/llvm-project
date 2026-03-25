// clang-format off
// RUN: %clangXX %flags %openmp_flags -fopenmp-version=60 %s -o %t && env KMP_TASKGRAPH_TRACE=1 %libomp-run 2>&1 | FileCheck %s

// REQUIRES: omp_taskgraph_experimental

int main()
{
  int deps[6];
  #pragma omp parallel
  {
    #pragma omp single
    {
      for (int i = 0; i < 2; i++)
      {
        #pragma omp taskgraph
        {
          #pragma omp task depend(out: deps[0], deps[1])
          { }
          #pragma omp task depend(out: deps[2], deps[3])
          { }
          #pragma omp task depend(inout: deps[0])
          { }
          #pragma omp task depend(inout: deps[1])
          { }
          #pragma omp task depend(inout: deps[2])
          { }
          #pragma omp task depend(inout: deps[3])
          { }
          #pragma omp task depend(in: deps[0], deps[1], deps[2], deps[3])
          { }
          #pragma omp task depend(in: deps[1], deps[2]) depend(out: deps[5])
          { }
          #pragma omp task depend(in: deps[5])
          { }
        }
      }
    }
  }
  return 0;
}

// This dependence graph is irreducible (it has no series-parallel
// decomposition), so the residual tangle is carved into a single
// TASKGRAPH_REGION_IRREDUCIBLE container whose children carry explicit intra
// edges.  The two tasks that form a clean producer/consumer chain on deps[5]
// (the depend(out: deps[5]) and depend(in: deps[5]) tasks) still reduce to a
// nested sequential.
// CHECK:      Processed taskgraph 0x[[#%x,GRAPHPTR:]] (graph_id 0):
// CHECK-NEXT: irreducible {
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   sequential {
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:     node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   }
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT:   node: 0x{{[[:xdigit:]]+}}
// CHECK-NEXT: }
// CHECK-NEXT: Replay taskgraph 0x[[#GRAPHPTR]] from task 0x{{[[:xdigit:]]+}}
