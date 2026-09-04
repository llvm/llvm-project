// RUN: mlir-opt %s -tosa-gather-scatter-hardening -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @gather(
// CHECK: %[[CLAMP:.*]] = tosa.clamp %arg1 {max_val = 20 : i32, min_val = 0 : i32} : (tensor<3x6xi32>) -> tensor<3x6xi32>
// CHECK: %[[OTHER_USE:.*]] = tosa.add %arg1, %arg2
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMP]]
// CHECK: return %[[GATHER]], %[[OTHER_USE]]
func.func @gather(%arg0: tensor<3x21x5xi8>, %arg1: tensor<3x6xi32>,
                  %arg2: tensor<3x6xi32>)
    -> (tensor<3x6x5xi8>, tensor<3x6xi32>) {
  %0 = tosa.add %arg1, %arg2 : (tensor<3x6xi32>, tensor<3x6xi32>) -> tensor<3x6xi32>
  %1 = tosa.gather %arg0, %arg1 : (tensor<3x21x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %1, %0 : tensor<3x6x5xi8>, tensor<3x6xi32>
}

// -----

// One clamp is shared by gather and scatter when their bounds match.

// CHECK-LABEL: func.func @shared_indices(
// CHECK: %[[CLAMP:.*]] = tosa.clamp %arg2 {max_val = 20 : i32, min_val = 0 : i32} : (tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK-NOT: tosa.clamp
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[CLAMP]]
// CHECK: %[[SCATTER:.*]] = tosa.scatter %arg1, %[[CLAMP]], %arg3
// CHECK-NOT: tosa.clamp
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

// Gather and scatter get separate clamps when their bounds differ.

// CHECK-LABEL: func.func @different_bounds(
// CHECK-DAG: %[[GATHER_CLAMP:.*]] = tosa.clamp %arg2 {max_val = 20 : i32, min_val = 0 : i32} : (tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK-DAG: %[[SCATTER_CLAMP:.*]] = tosa.clamp %arg2 {max_val = 51 : i32, min_val = 0 : i32} : (tensor<2x4xi32>) -> tensor<2x4xi32>
// CHECK: %[[GATHER:.*]] = tosa.gather %arg0, %[[GATHER_CLAMP]]
// CHECK: %[[SCATTER:.*]] = tosa.scatter %arg1, %[[SCATTER_CLAMP]], %arg3
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
// CHECK: %[[CLAMP:.*]] = tosa.clamp %arg1 {max_val = 26 : i64, min_val = 0 : i64} : (tensor<13x4xi64>) -> tensor<13x4xi64>
// CHECK: tosa.scatter %arg0, %[[CLAMP]], %arg2
func.func @i64_indices(%arg0: tensor<13x27x3xi16>,
                       %arg1: tensor<13x4xi64>,
                       %arg2: tensor<13x4x3xi16>) -> tensor<13x27x3xi16> {
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<13x27x3xi16>, tensor<13x4xi64>, tensor<13x4x3xi16>) -> tensor<13x27x3xi16>
  return %0 : tensor<13x27x3xi16>
}

// -----

// An i32 index cannot represent K - 1 here, so its signed maximum is a safe
// upper clamp bound.

// CHECK-LABEL: func.func @large_indexed_dimension(
// CHECK: %[[CLAMP:.*]] = tosa.clamp %arg1 {max_val = 2147483647 : i32, min_val = 0 : i32}
// CHECK: tosa.gather %arg0, %[[CLAMP]]
func.func @large_indexed_dimension(
    %arg0: tensor<1x2147483649x1xi8>, %arg1: tensor<1x1xi32>)
    -> tensor<1x1x1xi8> {
  %0 = tosa.gather %arg0, %arg1 : (tensor<1x2147483649x1xi8>, tensor<1x1xi32>) -> tensor<1x1x1xi8>
  return %0 : tensor<1x1x1xi8>
}

// -----

// CHECK-LABEL: func.func @already_hardened(
// CHECK: %[[CLAMP:.*]] = tosa.clamp %arg1 {max_val = 20 : i32, min_val = 0 : i32}
// CHECK-NOT: tosa.clamp
// CHECK: tosa.gather %arg0, %[[CLAMP]]
func.func @already_hardened(%arg0: tensor<3x21x5xi8>,
                            %arg1: tensor<3x6xi32>) -> tensor<3x6x5xi8> {
  %0 = tosa.clamp %arg1 {max_val = 20 : i32, min_val = 0 : i32} : (tensor<3x6xi32>) -> tensor<3x6xi32>
  %1 = tosa.gather %arg0, %0 : (tensor<3x21x5xi8>, tensor<3x6xi32>) -> tensor<3x6x5xi8>
  return %1 : tensor<3x6x5xi8>
}

// -----

// CHECK-LABEL: func.func @no_gather_or_scatter(
// CHECK-NOT: tosa.clamp
// CHECK: %[[ADD:.*]] = tosa.add %arg0, %arg1
// CHECK: return %[[ADD]]
func.func @no_gather_or_scatter(%arg0: tensor<2x4xi32>,
                                %arg1: tensor<2x4xi32>) -> tensor<2x4xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2x4xi32>, tensor<2x4xi32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
