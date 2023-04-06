// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK: #[[$map:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-LABEL: func @affine_apply(
//  CHECK-SAME:     %[[a:.*]]: index, %[[b:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]], %[[b]]]
//       CHECL:   return %[[apply]]
func.func @affine_apply(%a: index, %b: index) -> index {
  %0 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%a, %b]
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}
