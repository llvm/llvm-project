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

// CHECK: #[[$map:.*]] = affine_map<()[s0] -> ((s0 + 4) floordiv 5)>
// CHECK-LABEL: func @arith_ceildivsi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[apply:.*]] = affine.apply #[[$map]]()[%[[a]]]
//       CHECK:   return %[[apply]]
func.func @arith_ceildivsi(%a: index) -> index {
  %0 = arith.constant 5 : index
  %1 = arith.ceildivsi %a, %0 : index
  %2 = "test.reify_bound"(%1) : (index) -> (index)
  return %2 : index
}

// -----

func.func @arith_ceildivsi_non_pure(%a: index, %b: index) -> index {
  %0 = arith.ceildivsi %a, %b : index
  // Semi-affine expressions (such as "symbol * symbol") are not supported.
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_divui_constant()
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @arith_divui_constant() -> index {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = arith.divui %c7, %c5 : index
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----


// CHECK-LABEL: func @arith_divui_positive_lhs()
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]

func.func @arith_divui_positive_lhs() -> index {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %lhs = arith.maxsi %c7, %c1 : index
  %0 = arith.divui %lhs, %c5 : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}


// -----

// CHECK-LABEL: func @arith_divsi_negative_positive()
//       CHECK:   %[[cm1:.*]] = arith.constant -1 : index
//       CHECK:   return %[[cm1]]
func.func @arith_divsi_negative_positive() -> index {
  %cm7 = arith.constant -7 : index
  %c5 = arith.constant 5 : index
  %0 = arith.divsi %cm7, %c5 : index
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_divsi_negative_lhs()
//       CHECK:   %[[cm2:.*]] = arith.constant -2 : index
//       CHECK:   return %[[cm2]]
func.func @arith_divsi_negative_lhs() -> index {
  %c2 = arith.constant 2 : index
  %cm7 = arith.constant -7 : index
  %cm5 = arith.constant -5 : index
  %lhs = arith.minsi %cm5, %cm7 : index
  %0 = arith.divsi %lhs, %c2 : index
  %1 = "test.reify_bound"(%0) {type = "UB", constant}: (index) -> (index)
  return %1 : index
}

// -----

func.func @arith_divsi_unknown_lhs_constant_rhs(%a: index) {
  %c5 = arith.constant 5 : index
  %0 = arith.divsi %a, %c5 : index
  // expected-remark @below{{true}}
  "test.compare"(%0, %a) {
      cmp = "GE",
      rhs_map = affine_map<()[s0] -> (s0 floordiv 5)>
  } : (index, index) -> ()
  // expected-remark @below{{true}}
  "test.compare"(%0, %a) {
      cmp = "LE",
      rhs_map = affine_map<()[s0] -> (s0 ceildiv 5)>
  } : (index, index) -> ()
  return
}

// -----

// CHECK-LABEL: func @arith_remsi_positive_positive()
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c0]], %[[c5]]
func.func @arith_remsi_positive_positive() -> (index, index) {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = arith.remsi %c7, %c5 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  %2 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @arith_remsi_negative_positive()
//       CHECK:   %[[cm4:.*]] = arith.constant -4 : index
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[cm4]], %[[c1]]
func.func @arith_remsi_negative_positive() -> (index, index) {
  %cm7 = arith.constant -7 : index
  %c5 = arith.constant 5 : index
  %0 = arith.remsi %cm7, %c5 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  %2 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @arith_remsi_positive_negative()
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c0]], %[[c5]]
func.func @arith_remsi_positive_negative() -> (index, index) {
  %c7 = arith.constant 7 : index
  %cm5 = arith.constant -5 : index
  %0 = arith.remsi %c7, %cm5 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  %2 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @arith_remsi_negative_negative()
//       CHECK:   %[[cm4:.*]] = arith.constant -4 : index
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[cm4]], %[[c1]]
func.func @arith_remsi_negative_negative() -> (index, index) {
  %cm7 = arith.constant -7 : index
  %cm5 = arith.constant -5 : index
  %0 = arith.remsi %cm7, %cm5 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  %2 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @arith_remsi_positive_lhs_symbolic_positive_rhs(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[lb:.*]] = arith.constant 0 : index
//       CHECK:   return %[[lb]]
func.func @arith_remsi_positive_lhs_symbolic_positive_rhs(%a: index) -> index {
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %rhs = arith.maxsi %a, %c1 : index
  %0 = arith.remsi %c7, %rhs : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_remsi_negative_lhs_symbolic_positive_rhs(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[ub:.*]] = arith.constant 1 : index
//       CHECK:   return %[[ub]]
func.func @arith_remsi_negative_lhs_symbolic_positive_rhs(%a: index) -> index {
  %c1 = arith.constant 1 : index
  %cm7 = arith.constant -7 : index
  %rhs = arith.maxsi %a, %c1 : index
  %0 = arith.remsi %cm7, %rhs : index
  %1 = "test.reify_bound"(%0) {type = "UB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_remui_constant()
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[c5:.*]] = arith.constant 5 : index
//       CHECK:   return %[[c0]], %[[c5]]
func.func @arith_remui_constant() -> (index, index) {
  %c7 = arith.constant 7 : index
  %c5 = arith.constant 5 : index
  %0 = arith.remui %c7, %c5 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  %2 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @arith_remui_symbolic_dividend(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[ub:.*]] = arith.constant 5 : index
//       CHECK:   return %[[ub]]
func.func @arith_remui_symbolic_dividend(%a: index) -> index {
  %c5 = arith.constant 5 : index
  %0 = arith.remui %a, %c5 : index
  %1 = "test.reify_bound"(%0) {type = "UB", constant} : (index) -> (index)
  return %1 : index
}

// -----

func.func @arith_remui_unknown_divisor(%a: index, %b: index) -> index {
  %0 = arith.remui %a, %b : index
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {type = "UB", constant} : (index) -> (index)
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

// -----

// CHECK-LABEL: func @arith_minsi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[ub:.*]] = arith.constant 5 : index
//       CHECK:   return %[[ub]]
func.func @arith_minsi(%a: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.minsi %a, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1 : index
}

// -----

func.func @arith_minsi_lb(%a: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.minsi %a, %c4 : index
  // Signed min has no lower bound.
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxsi(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   arith.constant 4 : index
//       CHECK:   %[[lb:.*]] = arith.constant 4 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxsi(%a: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.maxsi %a, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  return %1 : index
}

// -----

func.func @arith_maxsi_ub(%a: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.maxsi %a, %c4 : index
  // Signed max has no upper bound.
  // expected-error @below{{could not reify bound}}
  %1 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_extsi_const(
//       CHECK:   %[[c:.*]] = arith.constant -5 : index
//       CHECK:   return %[[c]]
func.func @arith_extsi_const() -> index {
  %c_5 = arith.constant -5 : i32
  %ext = arith.extsi %c_5 : i32 to i64
  %0 = "test.reify_bound"(%ext) {constant, allow_integer_type} : (i64) -> (index)
  return %0 : index
}

// -----

// CHECK-LABEL: func @arith_minui(
//       CHECK:   %[[ub:.*]] = arith.constant 5 : index
//       CHECK:   return %[[ub]]
func.func @arith_minui() -> index {
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %0 = arith.minui %c10, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui(
//       CHECK:   %[[lb:.*]] = arith.constant 10 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui() -> index {
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %0 = arith.maxui %c10, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui_unknown_operand(
//       CHECK:   %[[lb:.*]] = arith.constant 4 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui_unknown_operand(%a: index) -> index {
  %c4 = arith.constant 4 : index
  %0 = arith.maxui %a, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_minui_wraparound(
//       CHECK:   %[[ub:.*]] = arith.constant 11 : index
//       CHECK:   return %[[ub]]
func.func @arith_minui_wraparound() -> index {
  %c255 = arith.constant 0xFF : i8
  %c10 = arith.constant 10 : i8
  %0 = arith.minui %c255, %c10 : i8
  %1 = "test.reify_bound"(%0) {type = "UB", allow_integer_type} : (i8) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui_wraparound(
//       CHECK:   %[[lb:.*]] = arith.constant 10 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui_wraparound() -> index {
  %c255 = arith.constant 0xFF : i8
  %c10 = arith.constant 10 : i8
  %0 = arith.maxui %c255, %c10 : i8
  %1 = "test.reify_bound"(%0) {type = "LB", allow_integer_type} : (i8) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui_addi(
//       CHECK:   %[[lb:.*]] = arith.constant 14 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui_addi() -> index {
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %sum = arith.addi %c4, %c10 : index
  %0 = arith.maxui %sum, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_minui_nonneg_symbolic(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[ub:.*]] = arith.constant 5 : index
//       CHECK:   return %[[ub]]
func.func @arith_minui_nonneg_symbolic(%a: index) -> index {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %nn = arith.maxsi %a, %c0 : index
  %0 = arith.minui %nn, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "UB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_minui_negative_symbolic(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[ub:.*]] = arith.constant 5 : index
//       CHECK:   return %[[ub]]
func.func @arith_minui_negative_symbolic(%a: index) -> index {
  %cm1 = arith.constant -1 : index
  %c4 = arith.constant 4 : index
  %neg = arith.minsi %a, %cm1 : index
  %0 = arith.minui %neg, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "UB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui_nonneg_symbolic(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[lb:.*]] = arith.constant 4 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui_nonneg_symbolic(%a: index) -> index {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %nn = arith.maxsi %a, %c0 : index
  %0 = arith.maxui %nn, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @arith_maxui_negative_symbolic(
//  CHECK-SAME:     %[[a:.*]]: index
//       CHECK:   %[[lb:.*]] = arith.constant 4 : index
//       CHECK:   return %[[lb]]
func.func @arith_maxui_negative_symbolic(%a: index) -> index {
  %cm1 = arith.constant -1 : index
  %c4 = arith.constant 4 : index
  %neg = arith.minsi %a, %cm1 : index
  %0 = arith.maxui %neg, %c4 : index
  %1 = "test.reify_bound"(%0) {type = "LB", constant} : (index) -> (index)
  return %1 : index
}
