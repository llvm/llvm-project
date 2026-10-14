// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-convert-integer-types='bitwidth-mappings=4:8,48:64,8:4' %s
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types='bitwidth-mappings=32:32' %s | FileCheck %s

// Unselected widths must leave even missing or malformed resources untouched.
// CHECK-LABEL: func.func @missing_resource()
// CHECK: tosa.const values(dense_resource<missing> : tensor<2xi48>) : () -> tensor<2xi48>

// A missing resource must produce a diagnostic, not an invalid i64 constant.
func.func @missing_resource() -> tensor<2xi48> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const'}}
  %0 = tosa.const values(dense_resource<missing> : tensor<2xi48>) : () -> tensor<2xi48>
  return %0 : tensor<2xi48>
}

// -----

// An i48 element occupies six bytes; this resource contains only two bytes.
func.func @malformed_resource() -> tensor<2xi48> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const'}}
  %0 = tosa.const values(dense_resource<malformed> : tensor<2xi48>) : () -> tensor<2xi48>
  return %0 : tensor<2xi48>
}
{-#
  dialect_resources: {
    builtin: {
      // 01000000: alignment 1. Payload: bytes 01 02, too short for an i48.
      malformed: "0x010000000102"
    }
  }
#-}

// -----

// Complete elements must match the element count of a static shape.
func.func @resource_wrong_element_count() -> tensor<2xi48> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const': failed to convert integer types or attributes}}
  %0 = tosa.const values(dense_resource<short> : tensor<2xi48>) : () -> tensor<2xi48>
  return %0 : tensor<2xi48>
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by one i48 value; the shape requires two.
  short: "0x01000000FFFFFFFFFFFF"
} } #-}

// -----

// An unresolved shape must not allow a payload containing an incomplete i48.
func.func @malformed_dynamic_resource() -> tensor<?xi48> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const': failed to convert integer types or attributes}}
  %0 = tosa.const values(dense_resource<partial> : tensor<?xi48>) : () -> tensor<?xi48>
  return %0 : tensor<?xi48>
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by only two payload bytes.
  partial: "0x010000000102"
} } #-}

// -----

// Report missing i48 resources nested in attributes even when the operation's
// operand and result types need no widening.
func.func @nested_missing_resource(%arg: tensor<2xi32>) -> tensor<2xi32> {
  // expected-error @+1 {{failed to convert integer types or attributes}}
  %0 = tosa.identity %arg {test.nested = [{value = dense_resource<missing> : tensor<2xi48>}]} : (tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @missing_i4_resource()
// CHECK: tosa.const values(dense_resource<missing> : tensor<2xi4>) : () -> tensor<2xi4>
func.func @missing_i4_resource() -> tensor<2xi4> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const': failed to convert integer types or attributes}}
  %0 = tosa.const values(dense_resource<missing> : tensor<2xi4>) : () -> tensor<2xi4>
  return %0 : tensor<2xi4>
}

// -----

// Each i4 requires one byte; a blob packing two nibbles per byte is too short.
func.func @packed_i4_resource() -> tensor<2xi4> {
  // expected-error @+1 {{failed to legalize operation 'tosa.const': failed to convert integer types or attributes}}
  %0 = tosa.const values(dense_resource<packed> : tensor<2xi4>) : () -> tensor<2xi4>
  return %0 : tensor<2xi4>
}
{-# dialect_resources: { builtin: {
  packed: "0x0100000079"
} } #-}

// -----

// Dense arrays cannot represent sub-byte destinations other than i1.
// expected-error @+1 {{failed to legalize operation 'builtin.module': failed to convert integer types or attributes}}
module attributes {test.array = array<i8: -1, 7>} {}
