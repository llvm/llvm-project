// RUN: mlir-opt -split-input-file -verify-diagnostics -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8,48:64' %s

// Complete elements must match the element count of a static shape.
// expected-error @+1 {{failed to convert integer types or attributes}}
module {
  func.func @resource_wrong_element_count() -> tensor<2xi48> {
    %0 = tosa.const values(dense_resource<short> : tensor<2xi48>) : () -> tensor<2xi48>
    return %0 : tensor<2xi48>
  }
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by one i48 value; the shape requires two.
  short: "0x01000000FFFFFFFFFFFF"
} } #-}

// -----

// An unresolved shape must not allow a payload containing an incomplete i48.
// expected-error @+1 {{failed to convert integer types or attributes}}
module {
  func.func @malformed_dynamic_resource() -> tensor<?xi48> {
    %0 = tosa.const values(dense_resource<partial> : tensor<?xi48>) : () -> tensor<?xi48>
    return %0 : tensor<?xi48>
  }
}
{-# dialect_resources: { builtin: {
  // Alignment 1 followed by only two payload bytes.
  partial: "0x010000000102"
} } #-}

// -----

// Report missing i48 resources nested in attributes even when the operation's
// operand and result types need no widening.
// expected-error @+1 {{failed to convert integer types or attributes}}
module {
  func.func @nested_missing_resource(%arg: tensor<2xi32>) -> tensor<2xi32> {
    %0 = tosa.identity %arg {test.nested = [{value = dense_resource<missing> : tensor<2xi48>}]} : (tensor<2xi32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

// Each i4 requires one byte; a blob packing two nibbles per byte is too short.
// expected-error @+1 {{failed to convert integer types or attributes}}
module {
  func.func @packed_i4_resource() -> tensor<2xi4> {
    %0 = tosa.const values(dense_resource<packed> : tensor<2xi4>) : () -> tensor<2xi4>
    return %0 : tensor<2xi4>
  }
}
{-# dialect_resources: { builtin: {
  packed: "0x0100000079"
} } #-}
