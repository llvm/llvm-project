// RUN: mlir-opt %s -tosa-gather-scatter-hardening -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @gather(
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK: %[[OTHER_USE:.*]] = tosa.add %arg1, %arg2
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK: return %[[GATHER]], %[[OTHER_USE]]
func.func @gather(%arg0: tensor<3x21x5xi8>, %arg1: tensor<3x6xi32>,
                  %arg2: tensor<3x6xi32>)
    -> (tensor<3x6x5xi8>, tensor<3x6xi32>) {
  %0 = tosa.add %arg1, %arg2 : (tensor<3x6xi32>, tensor<3x6xi32>) -> tensor<3x6xi32>
  %1 = tosa.gather %arg0, %arg1 : (tensor<3x21x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %1, %0 : tensor<3x6x5xi8>, tensor<3x6xi32>
}

// -----

// One bounding sequence is shared by gather and scatter when their bounds
// match.

// CHECK-LABEL: func.func @shared_indices(
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg2, %[[ZERO]]
// CHECK: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NOT: tosa.maximum
// CHECK-NOT: tosa.minimum
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK: %[[SCATTER:.*]] = tosa.scatter %arg1, %[[CLAMPED]], %arg3
// CHECK-NOT: tosa.maximum
// CHECK-NOT: tosa.minimum
// CHECK: return %[[GATHER]], %[[SCATTER]]
func.func @shared_indices(%arg0: tensor<2x21x3xf32>,
                          %arg1: tensor<2x21x3xf32>,
                          %arg2: tensor<2x4xi32>,
                          %arg3: tensor<2x4x3xf32>)
    -> (tensor<2x4x3xf32>, tensor<2x21x3xf32>) {
  %0 = tosa.gather %arg0, %arg2 : (tensor<2x21x3xf32>, tensor<2x4xi32>) -> tensor<2x4x3xf32>
  %1 = tosa.scatter %arg1, %arg2, %arg3 : (tensor<2x21x3xf32>, tensor<2x4xi32>, tensor<2x4x3xf32>) -> tensor<2x21x3xf32>
  return %0, %1 : tensor<2x4x3xf32>, tensor<2x21x3xf32>
}

// -----

// Gather and scatter get separate bounding sequences when their bounds differ.

// CHECK-LABEL: func.func @different_bounds(
// CHECK: %[[SCATTER_ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[SCATTER_UPPER:.*]] = "tosa.const"() <{values = dense<51> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[SCATTER_NONNEGATIVE:.*]] = tosa.maximum %arg2, %[[SCATTER_ZERO]]
// CHECK: %[[SCATTER_CLAMPED:.*]] = tosa.minimum %[[SCATTER_NONNEGATIVE]], %[[SCATTER_UPPER]]
// CHECK: %[[GATHER_ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[GATHER_UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[GATHER_NONNEGATIVE:.*]] = tosa.maximum %arg2, %[[GATHER_ZERO]]
// CHECK: %[[GATHER_CLAMPED:.*]] = tosa.minimum %[[GATHER_NONNEGATIVE]], %[[GATHER_UPPER]]
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[GATHER_CLAMPED]]
// CHECK: %[[SCATTER:.*]] = tosa.scatter %arg1, %[[SCATTER_CLAMPED]], %arg3
// CHECK: return %[[GATHER]], %[[SCATTER]]
func.func @different_bounds(%arg0: tensor<2x21x3xf32>,
                            %arg1: tensor<2x52x3xf32>,
                            %arg2: tensor<2x4xi32>,
                            %arg3: tensor<2x4x3xf32>)
    -> (tensor<2x4x3xf32>, tensor<2x52x3xf32>) {
  %0 = tosa.gather %arg0, %arg2 : (tensor<2x21x3xf32>, tensor<2x4xi32>) -> tensor<2x4x3xf32>
  %1 = tosa.scatter %arg1, %arg2, %arg3 : (tensor<2x52x3xf32>, tensor<2x4xi32>, tensor<2x4x3xf32>) -> tensor<2x52x3xf32>
  return %0, %1 : tensor<2x4x3xf32>, tensor<2x52x3xf32>
}

// -----

// CHECK-LABEL: func.func @i64_indices(
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK: %[[UPPER:.*]] = "tosa.const"() <{values = dense<26> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK: tosa.scatter %arg0, %[[CLAMPED]], %arg2
func.func @i64_indices(%arg0: tensor<13x27x3xi16>,
                       %arg1: tensor<13x4xi64>,
                       %arg2: tensor<13x4x3xi16>) -> tensor<13x27x3xi16> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x27x3xi16>, tensor<13x4xi64>, tensor<13x4x3xi16>) -> tensor<13x27x3xi16>
  return %0 : tensor<13x27x3xi16>
}

// -----

// An i32 index cannot represent K - 1 here, so its signed maximum is a safe
// upper bound.

// CHECK-LABEL: func.func @large_indexed_dimension(
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[UPPER:.*]] = "tosa.const"() <{values = dense<2147483647> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK: tosa.gather %arg0, %[[CLAMPED]]
func.func @large_indexed_dimension(
    %arg0: tensor<1x2147483649x1xi8>, %arg1: tensor<1x1xi32>)
    -> tensor<1x1x1xi8> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<1x2147483649x1xi8>, tensor<1x1xi32>) -> tensor<1x1x1xi8>
  return %0 : tensor<1x1x1xi8>
}

// -----

// CHECK-LABEL: func.func @already_hardened(
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NOT: tosa.maximum
// CHECK-NOT: tosa.minimum
// CHECK: tosa.gather %arg0, %[[CLAMPED]]
func.func @already_hardened(%arg0: tensor<3x21x5xi8>,
                            %arg1: tensor<3x6xi32>) -> tensor<3x6x5xi8> {
  %0 = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %1 = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %2 = tosa.maximum %arg1, %0 : (tensor<3x6xi32>, tensor<1x1xi32>) -> tensor<3x6xi32>
  %3 = tosa.minimum %2, %1 : (tensor<3x6xi32>, tensor<1x1xi32>) -> tensor<3x6xi32>
  %4 = tosa.gather %arg0, %3 : (tensor<3x21x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %4 : tensor<3x6x5xi8>
}

// -----

// CHECK-LABEL: func.func @no_gather_or_scatter(
// CHECK-NOT: tosa.const
// CHECK-NOT: tosa.maximum
// CHECK-NOT: tosa.minimum
// CHECK: %[[ADD:.*]] = tosa.add %arg0, %arg1
// CHECK: return %[[ADD]]
func.func @no_gather_or_scatter(%arg0: tensor<2x4xi32>,
                                %arg1: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
