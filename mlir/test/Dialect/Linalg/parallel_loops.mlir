// RUN: mlir-opt %s -convert-linalg-to-parallel-loops -split-input-file | FileCheck %s --dump-input-on-failure

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @linalg_generic_sum(%lhs: memref<2x2xf32>,
                         %rhs: memref<2x2xf32>,
                         %sum: memref<2x2xf32>) {
  linalg.generic {
    args_in = 2 : i64,
    args_out = 1 : i64,
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"]
  } %lhs, %rhs, %sum {
    ^bb0(%lhs_in: f32, %rhs_in: f32, %sum_out: f32):   // no predecessors
      %0 = addf %lhs_in, %rhs_in : f32
      linalg.yield %0 : f32
  }: memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>
  return
}
// CHECK-LABEL: @linalg_generic_sum
// CHECK:   (%[[LHS:.*]]:{{.*}}, %[[RHS:.*]]:{{.*}}, %[[SUM:.*]]:{{.*}})
// CHECK-DAG: %[[C2:.*]] = constant 2
// CHECK-DAG: %[[C0:.*]] = constant 0
// CHECK-DAG: %[[C1:.*]] = constant 1
// CHECK: loop.parallel (%[[I:.*]], %[[J:.*]]) = {{.*}}
// CHECK:   %[[LHS_ELEM:.*]] = load %[[LHS]][%[[I]], %[[J]]]
// CHECK:   %[[RHS_ELEM:.*]] = load %[[RHS]][%[[I]], %[[J]]]
// CHECK:   %[[SUM_ELEM:.*]] = load %[[SUM]][%[[I]], %[[J]]]
// CHECK:   %[[SUM:.*]] = addf %[[LHS_ELEM]], %[[RHS_ELEM]] : f32
// CHECK:   store %[[SUM]], %{{.*}}[%[[I]], %[[J]]]
// CHECK:   loop.yield

// -----

#accesses = [
  affine_map<(m, n) -> (m, n)>,
  affine_map<(m, n) -> (m)>
]
#trait = {
  args_in = 1,
  args_out = 1,
  iterator_types = ["parallel", "reduction"],
  indexing_maps = #accesses
}

func @do_not_lower_reduce(%A: memref<2x4xf32>, %B: memref<2xf32>) {
  linalg.generic #trait %A, %B {
    ^bb0(%a: f32, %b: f32):
      linalg.yield %a: f32
  } : memref<2x4xf32>, memref<2xf32>
  return
}
// CHECK-LABEL: @do_not_lower_reduce
// CHECK: linalg.generic
