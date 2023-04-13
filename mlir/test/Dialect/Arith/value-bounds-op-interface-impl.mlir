// RUN: mlir-opt %s -test-affine-reify-value-bounds -verify-diagnostics \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-LABEL: func @arith_addi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_addi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.addi %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (-s0 + 5)>
// CHECK-LABEL: func @arith_subi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_subi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.subi %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 * 5)>
// CHECK-LABEL: func @arith_muli(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_muli(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.muli %0, %a : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

func.func @arith_muli_non_pure(%a: index, %b: index) -> index {
  %0 = arith.muli %a, %b : index
  // Semi-affine expressions (such as "symbol * symbol") are not supported.
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_const()
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c5]]
func.func @arith_const() -> index {
  %c5 = arith.constant 5 : index
  %0 = "test.reify_bound"(%c5) : (index) -> (index)
  return %0 : index
}
