// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -verify-diagnostics -split-input-file | FileCheck %s

// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds{use-arith-ops}))' \
// RUN:     -verify-diagnostics -split-input-file | \
// RUN: FileCheck %s --check-prefix=CHECK-ARITH

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-LABEL: func @arith_addi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]

// CHECK-ARITH-LABEL: func @arith_addi(
//  CHECK-ARITH-SAME:     %[[a:.*]]: index
//       CHECK-ARITH:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK-ARITH:   %[[add:.*]] = arith.addi %[[c5]], %[[a]]
//       CHECK-ARITH:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK-ARITH:   %[[add:.*]] = arith.addi %[[a]], %[[c5]]
//       CHECK-ARITH:   return %[[add]]
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

// CHECK: #[[$map_muli_i32:.*]] = affine_map<()[s0] -> (s0 * 7)>
// CHECK-LABEL: func @arith_muli_integer(
//  CHECK-SAME:     %[[a:.*]]: i32
//       CHECK:   %[[cast:.*]] = arith.index_cast %[[a]] : i32 to index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map_muli_i32]]()[%[[cast]]]
//       CHECK:   return %[[apply]]
// CHECK-ARITH-LABEL: func @arith_muli_integer(
//  CHECK-ARITH-SAME:     %[[a:.*]]: i32
//       CHECK-ARITH:   %[[c7:.*]] = arith.constant 7 : i32
//       CHECK-ARITH:   arith.muli %[[a]], %[[c7]] : i32
//   CHECK-ARITH-DAG:   %[[cast:.*]] = arith.index_cast %[[a]] : i32 to index
//   CHECK-ARITH-DAG:   %[[c7_reified:.*]] = arith.constant 7 : index
//       CHECK-ARITH:   %[[mul:.*]] = arith.muli %[[cast]], %[[c7_reified]] : index
//       CHECK-ARITH:   return %[[mul]]
func.func @arith_muli_integer(%a: i32) -> index {
  %c7 = arith.constant 7 : i32
  %product = arith.muli %a, %c7 : i32
  %0 = "test.reify_bound"(%product) {allow_integer_type} : (i32) -> (index)
  return %0 : index
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

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> (s0 floordiv 5)>
// CHECK-LABEL: func @arith_floordivsi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_floordivsi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.floordivsi %a, %0 : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

func.func @arith_floordivsi_non_pure(%a: index, %b: index) -> index {
  %0 = arith.floordivsi %a, %b : index
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

// -----

// CHECK-LABEL: func @arith_addi_integer_constant()
//       CHECK:   %[[c12:.*]] = arith.constant 12 : index
//       CHECK:   return %[[c12]]
// CHECK-ARITH-LABEL: func @arith_addi_integer_constant()
//       CHECK-ARITH:   %[[c12:.*]] = arith.constant 12 : index
//       CHECK-ARITH:   return %[[c12]]
func.func @arith_addi_integer_constant() -> index {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %sum = arith.addi %c5, %c7 : i32
  %0 = "test.reify_bound"(%sum) {allow_integer_type, constant} : (i32) -> (index)
  return %0 : index
}

// -----

// CHECK-LABEL: func @arith_select(
func.func @arith_select(%c: i1) -> (index, index) {
  // CHECK: arith.constant 5 : index
  %c5 = arith.constant 5 : index
  // CHECK: arith.constant 9 : index
  %c9 = arith.constant 9 : index
  %r = arith.select %c, %c5, %c9 : index
  // CHECK: %[[c5:.*]] = arith.constant 5 : index
  // CHECK: %[[c10:.*]] = arith.constant 10 : index
  %0 = "test.reify_bound"(%r) {type = "LB"} : (index) -> (index)
  %1 = "test.reify_bound"(%r) {type = "UB"} : (index) -> (index)
  // CHECK: return %[[c5]], %[[c10]]
  return %0, %1 : index, index
}

// -----

// CHECK-LABEL: func @arith_select_elementwise(
//  CHECK-SAME:     %[[a:.*]]: tensor<?xf32>, %[[b:.*]]: tensor<?xf32>, %[[c:.*]]: tensor<?xi1>)
func.func @arith_select_elementwise(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<?xi1>) -> index {
  %r = arith.select %c, %a, %b : tensor<?xi1>, tensor<?xf32>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[dim:.*]] = tensor.dim %[[a]], %[[c0]]
  %0 = "test.reify_bound"(%r) {type = "EQ", dim = 0}
      : (tensor<?xf32>) -> (index)
  // CHECK: return %[[dim]]
  return %0 : index
}
