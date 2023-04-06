// RUN: mlir-opt %s -test-affine-reify-value-bounds="reify-to-func-args" \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: func @reify_through_chain(
//  CHECK-SAME:     %[[sz0:.*]]: index, %[[sz2:.*]]: index
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[sz0]], %[[c10]], %[[sz2]]
func.func @reify_through_chain(%sz0: index, %sz2: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %0 = tensor.empty(%sz0, %sz2) : tensor<?x10x?xf32>
  %1 = tensor.cast %0 : tensor<?x10x?xf32> to tensor<?x?x?xf32>
  %pos = arith.constant 0 : index
  %f = arith.constant 0.0 : f32
  %2 = tensor.insert %f into %1[%pos, %pos, %pos] : tensor<?x?x?xf32>
  %3 = tensor.dim %2, %c2 : tensor<?x?x?xf32>

  %4 = "test.reify_bound"(%2) {dim = 0} : (tensor<?x?x?xf32>) -> (index)
  %5 = "test.reify_bound"(%2) {dim = 1} : (tensor<?x?x?xf32>) -> (index)
  %6 = "test.reify_bound"(%3) : (index) -> (index)

  return %4, %5, %6 : index, index, index
}
