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

// -----

// CHECK-LABEL: func @affine_max_lb(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK:   return %[[c2]]
func.func @affine_max_lb(%a: index) -> (index) {
  // Note: There are two LBs: s0 and 2. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_max_ub(%a: index) -> (index) {
  %1 = affine.max affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

// CHECK-LABEL: func @affine_min_ub(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   return %[[c3]]
func.func @affine_min_ub(%a: index) -> (index) {
  // Note: There are two UBs: s0 + 1 and 3. FlatAffineValueConstraints always
  // returns the constant one at the moment.
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  %2 = "test.reify_bound"(%1) {type = "UB"}: (index) -> (index)
  return %2 : index
}

// -----

func.func @affine_min_lb(%a: index) -> (index) {
  %1 = affine.min affine_map<()[s0] -> (s0, 2)>()[%a]
  // expected-error @below{{could not reify bound}}
  %2 = "test.reify_bound"(%1) {type = "LB"}: (index) -> (index)
  return %2 : index
}
