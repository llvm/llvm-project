// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @linalg_fill(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @linalg_fill(%t: tensor<?xf32>, %f: f32) -> index {
  %0 = linalg.fill ins(%f : f32) outs(%t : tensor<?xf32>) -> tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}
