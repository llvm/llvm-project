// RUN: mlir-opt %s -tosa-gather-scatter-hardening -split-input-file | FileCheck %s
// RUN: mlir-opt %s -tosa-gather-scatter-hardening -tosa-gather-scatter-hardening -split-input-file | FileCheck %s

// Add a broadcastable clamp without changing other uses of the indices.
// CHECK-LABEL: func.func @gather(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]], %arg1
func.func @gather(%values: tensor<3x21x5xi8>, %indices: tensor<3x6xi32>)
    -> (tensor<3x6x5xi8>, tensor<3x6xi32>) {
  %0 = tosa.gather %values, %indices : (tensor<3x21x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %0, %indices : tensor<3x6x5xi8>, tensor<3x6xi32>
}

// -----

// Recognize an existing maximum followed by minimum, with commuted operands.
// CHECK-LABEL: func.func @already_hardened_maximum_first(
// CHECK: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1,
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]],
// CHECK-NEXT: %[[SCATTER:.*]] = tosa.scatter %arg0, %[[CLAMPED]], %arg2
// CHECK-NEXT: return %[[SCATTER]]
func.func @already_hardened_maximum_first(
    %values: tensor<1x21x1xi8>, %indices: tensor<1x2xi32>,
    %updates: tensor<1x2x1xi8>) -> tensor<1x21x1xi8> {
  %zero = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.maximum %zero, %indices : (tensor<1x1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  %1 = tosa.minimum %upper, %0 : (tensor<1x1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  %2 = tosa.scatter %values, %1, %updates : (tensor<1x21x1xi8>, tensor<1x2xi32>, tensor<1x2x1xi8>) -> tensor<1x21x1xi8>
  return %2 : tensor<1x21x1xi8>
}

// -----

// Recognize minimum followed by maximum. Both inner operands are constants;
// the data splat (25) must not hide the valid upper bound (20).
// CHECK-LABEL: func.func @already_hardened_minimum_first(
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-DAG: %[[DATA:.*]] = "tosa.const"() <{values = dense<25> : tensor<1x2xi32>}>
// CHECK: %[[BOUNDED:.*]] = tosa.minimum %[[UPPER]], %[[DATA]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.maximum %[[BOUNDED]],
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @already_hardened_minimum_first(%values: tensor<1x21x1xi8>)
    -> tensor<1x2x1xi8> {
  %zero = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %data = "tosa.const"() <{values = dense<25> : tensor<1x2xi32>}> : () -> tensor<1x2xi32>
  %0 = tosa.minimum %upper, %data : (tensor<1x1xi32>, tensor<1x2xi32>) -> tensor<1x2xi32>
  %1 = tosa.maximum %0, %zero : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %2 = tosa.gather %values, %1 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %2 : tensor<1x2x1xi8>
}

// -----

// No inner op is needed: minimum(25, 20) produces 20, within [0, 20].
// The first constant is out of bounds, so the second must be tried.
// CHECK-LABEL: func.func @constant_indices(
// CHECK-DAG: %[[DATA:.*]] = "tosa.const"() <{values = dense<25> : tensor<1x2xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-NEXT: %[[BOUNDED:.*]] = tosa.minimum %[[DATA]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[BOUNDED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @constant_indices(%values: tensor<1x21x1xi8>) -> tensor<1x2x1xi8> {
  %data = "tosa.const"() <{values = dense<25> : tensor<1x2xi32>}> : () -> tensor<1x2xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.minimum %data, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.gather %values, %0 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %1 : tensor<1x2x1xi8>
}

// -----

// Cap the upper bound at the largest representable index.
// CHECK-LABEL: func.func @large_indexed_dimension(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<2147483647> : tensor<1x1xi32>}>
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @large_indexed_dimension(
    %values: tensor<1x2147483649x1xi8>, %indices: tensor<1x1xi32>)
    -> tensor<1x1x1xi8> {
  %0 = tosa.gather %values, %indices : (tensor<1x2147483649x1xi8>, tensor<1x1xi32>) -> tensor<1x1x1xi8>
  return %0 : tensor<1x1x1xi8>
}

// -----

// An outer maximum can override a valid upper bound. Add a new clamp using
// the i64 index type rather than accepting the existing [21, 21] range.
// CHECK-LABEL: func.func @wrong_bound(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi64>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi64>}>
// CHECK-DAG: %[[BAD_BOUND:.*]] = "tosa.const"() <{values = dense<21> : tensor<1x1xi64>}>
// CHECK-NEXT: %[[BOUNDED:.*]] = tosa.minimum %arg1, %[[UPPER]]
// CHECK-NEXT: %[[ORIGINAL:.*]] = tosa.maximum %[[BOUNDED]], %[[BAD_BOUND]]
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %[[ORIGINAL]], %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[SCATTER:.*]] = tosa.scatter %arg0, %[[CLAMPED]], %arg2
// CHECK-NEXT: return %[[SCATTER]]
func.func @wrong_bound(
    %values: tensor<1x21x1xi8>, %indices: tensor<1x2xi64>,
    %updates: tensor<1x2x1xi8>) -> tensor<1x21x1xi8> {
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
  %bad_bound = "tosa.const"() <{values = dense<21> : tensor<1x1xi64>}> : () -> tensor<1x1xi64>
  %0 = tosa.minimum %indices, %upper : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
  %1 = tosa.maximum %0, %bad_bound : (tensor<1x2xi64>, tensor<1x1xi64>) -> tensor<1x2xi64>
  %2 = tosa.scatter %values, %1, %updates : (tensor<1x21x1xi8>, tensor<1x2xi64>, tensor<1x2x1xi8>) -> tensor<1x21x1xi8>
  return %2 : tensor<1x21x1xi8>
}

// -----

// A valid outer upper bound does not repair a negative inner lower bound.
// CHECK-LABEL: func.func @invalid_inner_bound(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK: %[[INNER:.*]] = tosa.maximum %arg1,
// CHECK-NEXT: %[[ORIGINAL:.*]] = tosa.minimum %[[INNER]], %[[UPPER]]
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %[[ORIGINAL]], %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @invalid_inner_bound(%values: tensor<1x21x1xi8>, %indices: tensor<1x2xi32>)
    -> tensor<1x2x1xi8> {
  %lower = "tosa.const"() <{values = dense<-1> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.maximum %indices, %lower : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.minimum %0, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %2 = tosa.gather %values, %1 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %2 : tensor<1x2x1xi8>
}

// -----

// A negative outer minimum overrides the valid inner lower bound.
// CHECK-LABEL: func.func @negative_outer_bound(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-DAG: %[[NEGATIVE:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x1xi32>}>
// CHECK-NEXT: %[[INNER:.*]] = tosa.maximum %arg1, %[[ZERO]]
// CHECK-NEXT: %[[ORIGINAL:.*]] = tosa.minimum %[[INNER]], %[[NEGATIVE]]
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %[[ORIGINAL]], %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @negative_outer_bound(%values: tensor<1x21x1xi8>, %indices: tensor<1x2xi32>)
    -> tensor<1x2x1xi8> {
  %zero = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %upper = "tosa.const"() <{values = dense<-1> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.maximum %indices, %zero : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.minimum %0, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %2 = tosa.gather %values, %1 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %2 : tensor<1x2x1xi8>
}

// -----

// An upper bound alone does not prove that indices are nonnegative.
// CHECK-LABEL: func.func @incomplete_clamp(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-NEXT: %[[ORIGINAL:.*]] = tosa.minimum %arg1, %[[UPPER]]
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %[[ORIGINAL]], %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @incomplete_clamp(%values: tensor<1x21x1xi8>, %indices: tensor<1x2xi32>)
    -> tensor<1x2x1xi8> {
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.minimum %indices, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.gather %values, %0 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %1 : tensor<1x2x1xi8>
}

// -----

// With two constants, one in-range operand is insufficient: minimum(-1, 20)
// produces -1 and still needs hardening.
// CHECK-LABEL: func.func @unsafe_constant_indices(
// CHECK-DAG: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1xi32>}>
// CHECK-DAG: %[[UPPER:.*]] = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}>
// CHECK-DAG: %[[DATA:.*]] = "tosa.const"() <{values = dense<-1> : tensor<1x2xi32>}>
// CHECK-NEXT: %[[ORIGINAL:.*]] = tosa.minimum %[[DATA]], %[[UPPER]]
// CHECK-NEXT: %[[NONNEGATIVE:.*]] = tosa.maximum %[[ORIGINAL]], %[[ZERO]]
// CHECK-NEXT: %[[CLAMPED:.*]] = tosa.minimum %[[NONNEGATIVE]], %[[UPPER]]
// CHECK-NEXT: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMPED]]
// CHECK-NEXT: return %[[GATHER]]
func.func @unsafe_constant_indices(%values: tensor<1x21x1xi8>) -> tensor<1x2x1xi8> {
  %data = "tosa.const"() <{values = dense<-1> : tensor<1x2xi32>}> : () -> tensor<1x2xi32>
  %upper = "tosa.const"() <{values = dense<20> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
  %0 = tosa.minimum %data, %upper : (tensor<1x2xi32>, tensor<1x1xi32>) -> tensor<1x2xi32>
  %1 = tosa.gather %values, %0 : (tensor<1x21x1xi8>, tensor<1x2xi32>) -> tensor<1x2x1xi8>
  return %1 : tensor<1x2x1xi8>
}
