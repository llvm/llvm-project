// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

func.func @positive_out_of_range() -> index {
  %wide = arith.constant 1180591620717411303424 : i128
  // expected-error @+1 {{could not reify bound}}
  %bound = "test.reify_bound"(%wide) {allow_integer_type, constant} : (i128) -> index
  return %bound : index
}

// -----

func.func @negative_out_of_range_operand(%x: i128) -> index {
  %wide = arith.constant -18446744073709551611 : i128
  %sum = arith.addi %x, %wide : i128
  // expected-error @+1 {{could not reify bound}}
  %bound = "test.reify_bound"(%sum) {allow_integer_type, constant} : (i128) -> index
  return %bound : index
}

// -----

// CHECK-LABEL: func.func @representable_wide_constant
// CHECK: %[[BOUND:.*]] = arith.constant -5 : index
// CHECK: return %[[BOUND]] : index
func.func @representable_wide_constant() -> index {
  %wide = arith.constant -5 : i128
  %bound = "test.reify_bound"(%wide) {allow_integer_type, constant} : (i128) -> index
  return %bound : index
}
