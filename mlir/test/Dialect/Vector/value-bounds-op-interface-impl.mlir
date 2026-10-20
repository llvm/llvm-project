// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @vector_transfer_write(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @vector_transfer_write(%t: tensor<?xf32>, %v: vector<5xf32>, %pos: index) -> index {
  %0 = vector.transfer_write %v, %t[%pos] : vector<5xf32>, tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}
