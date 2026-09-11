// RUN: mlir-opt %s --transform-interpreter=entry-point=width64 | FileCheck %s --check-prefixes=CHECK,WIDTH64
// RUN: mlir-opt %s --transform-interpreter=entry-point=skip | FileCheck %s --check-prefixes=CHECK,SKIP
// RUN: mlir-opt %s --transform-interpreter=entry-point=width8 | FileCheck %s --check-prefixes=CHECK,WIDTH8

// `index`-typed steps are materialized with the requested index bitwidth; a
// bitwidth of 0 leaves them untouched. Integer steps are unaffected by the
// option.

// CHECK-LABEL: @step_index
// WIDTH64: arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// SKIP: vector.step : vector<4xindex>
// WIDTH8: arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
func.func @step_index() -> vector<4xindex> {
  %0 = vector.step : vector<4xindex>
  return %0 : vector<4xindex>
}

// The lane values only wrap when the index bitwidth is small enough. With
// `index_bitwidth = 8` the sequence wraps at 256, so element 256 becomes 0 and
// element 257 becomes 1; with the 64-bit index it does not wrap (256 and 257
// are represented as-is).

// CHECK-LABEL: @step_index_wrap
// WIDTH64: arith.constant dense<"0x0000000000000000{{.*}}00010000000000000101000000000000"> : vector<258xindex>
// SKIP: vector.step : vector<258xindex>
// WIDTH8: arith.constant dense<"0x0000000000000000{{.*}}00000000000000000100000000000000"> : vector<258xindex>
func.func @step_index_wrap() -> vector<258xindex> {
  %0 = vector.step : vector<258xindex>
  return %0 : vector<258xindex>
}

// Integer steps are independent of the index bitwidth.
// CHECK-LABEL: @step_i8
// CHECK: arith.constant dense<[0, 1, 2, 3]> : vector<4xi8>
func.func @step_i8() -> vector<4xi8> {
  %0 = vector.step : vector<4xi8>
  return %0 : vector<4xi8>
}

// Negative test: scalable steps are never materialized as a constant - the
// pattern bails out regardless of the requested index bitwidth.
// CHECK-LABEL: @step_scalable_index
// CHECK-NOT: arith.constant
// CHECK: vector.step : vector<[4]xindex>
func.func @step_scalable_index() -> vector<[4]xindex> {
  %0 = vector.step : vector<[4]xindex>
  return %0 : vector<[4]xindex>
}

// Negative test: scalability is checked before the element type, so scalable
// integer steps are left untouched as well.
// CHECK-LABEL: @step_scalable_i8
// CHECK-NOT: arith.constant
// CHECK: vector.step : vector<[4]xi8>
func.func @step_scalable_i8() -> vector<[4]xi8> {
  %0 = vector.step : vector<[4]xi8>
  return %0 : vector<[4]xi8>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @width64(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_step index_bitwidth = 64
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @skip(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_step index_bitwidth = 0
    } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @width8(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.lower_step index_bitwidth = 8
    } : !transform.op<"func.func">
    transform.yield
  }
}
