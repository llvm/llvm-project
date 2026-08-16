// RUN: tr-opt %s --convert-tr-to-linalg | FileCheck %s

// Milestone 8: row / column / full reductions become linalg.generic over
// MemRefs. The scalar combiner is arith.addf. No Tensor dialect.

// CHECK-DAG: #[[IN:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[ROW:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[COL:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[FULL:.*]] = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: func.func @row
// CHECK-SAME: (%[[T:.*]]: memref<128x128xf32>) -> memref<128xf32>
func.func @row(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // CHECK: %[[ACC:.*]] = memref.alloca() : memref<128xf32>
  // CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: linalg.fill ins(%[[Z]] : f32) outs(%[[ACC]] : memref<128xf32>)
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[IN]], #[[ROW]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK-SAME: ins(%[[T]] : memref<128x128xf32>)
  // CHECK-SAME: outs(%[[ACC]] : memref<128xf32>)
  // CHECK: arith.addf
  // CHECK: linalg.yield
  // CHECK: return %[[ACC]] : memref<128xf32>
  %r = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @column
// CHECK-SAME: (%[[T:.*]]: memref<128x128xf32>) -> memref<128xf32>
func.func @column(%t: !tr.tile<128x128xf32>) -> !tr.tile<128xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[IN]], #[[COL]]]
  // CHECK-SAME: iterator_types = ["reduction", "parallel"]
  // CHECK: arith.addf
  %r = tr.reduce_sum %t, axis = 0 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @full
// CHECK-SAME: (%[[T:.*]]: memref<128x128xf32>) -> memref<f32>
func.func @full(%t: !tr.tile<128x128xf32>) -> !tr.tile<f32> {
  // Two successive reduces fuse to one generic over both axes.
  // CHECK: %[[ACC:.*]] = memref.alloca() : memref<f32>
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#[[IN]], #[[FULL]]]
  // CHECK-SAME: iterator_types = ["reduction", "reduction"]
  // CHECK-SAME: ins(%[[T]] : memref<128x128xf32>)
  // CHECK-SAME: outs(%[[ACC]] : memref<f32>)
  // CHECK: arith.addf
  // CHECK-NOT: tr.reduce_sum
  %r0 = tr.reduce_sum %t, axis = 1 : !tr.tile<128x128xf32> -> !tr.tile<128xf32>
  %r1 = tr.reduce_sum %r0, axis = 0 : !tr.tile<128xf32> -> !tr.tile<f32>
  return %r1 : !tr.tile<f32>
}

// CHECK-LABEL: func.func @add_tiles
// CHECK-SAME: (%[[A:.*]]: memref<128xf32>, %[[B:.*]]: memref<128xf32>) -> memref<128xf32>
func.func @add_tiles(%a: !tr.tile<128xf32>, %b: !tr.tile<128xf32>) -> !tr.tile<128xf32> {
  // CHECK: %[[OUT:.*]] = memref.alloca() : memref<128xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK-SAME: ins(%[[A]], %[[B]] : memref<128xf32>, memref<128xf32>)
  // CHECK-SAME: outs(%[[OUT]] : memref<128xf32>)
  // CHECK: arith.addf
  %r = tr.add %a, %b : !tr.tile<128xf32>
  return %r : !tr.tile<128xf32>
}

// CHECK-LABEL: func.func @constant_tile
func.func @constant_tile() -> !tr.tile<128xf32> {
  // CHECK: %[[A:.*]] = memref.alloca() : memref<128xf32>
  // CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: linalg.fill ins(%[[Z]] : f32) outs(%[[A]] : memref<128xf32>)
  // CHECK: return %[[A]] : memref<128xf32>
  %z = tr.constant 0.0 : !tr.tile<128xf32>
  return %z : !tr.tile<128xf32>
}
