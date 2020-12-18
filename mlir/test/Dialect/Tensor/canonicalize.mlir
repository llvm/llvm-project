// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s

// Checks that NOP casts are removed.
// CHECK-LABEL: cast_values
func @cast_values(%arg0: tensor<*xi32>) -> tensor<2xi32> {
  // NOP cast
  %0 = tensor.cast %arg0 : tensor<*xi32> to tensor<*xi32>
  // CHECK-NEXT: %[[RET:.*]] = tensor.cast %arg0 : tensor<*xi32> to tensor<2xi32>
  %2 = tensor.cast %0 : tensor<*xi32> to tensor<2xi32>
  // NOP cast
  %4 = tensor.cast %2 : tensor<2xi32> to tensor<2xi32>
  // CHECK-NEXT: return %[[RET]] : tensor<2xi32>
  return %4 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_ok
// CHECK-SAME: %[[IN:.*]]: tensor<*xi32>
func @tensor.cast_chain_ok(%input: tensor<*xi32>) -> tensor<4x8xi32> {
  // CHECK-NEXT: %[[RES:.*]] = tensor.cast %[[IN]] : tensor<*xi32> to tensor<4x8xi32>
  %0 = tensor.cast %input : tensor<*xi32> to tensor<4x?xi32>
  %1 = tensor.cast %0 : tensor<4x?xi32> to tensor<4x8xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<4x8xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_regain
// CHECK-SAME: %[[IN:.*]]: tensor<4xi32>
func @tensor.cast_chain_regain(%input: tensor<4xi32>) -> tensor<4xi32> {
  %0 = tensor.cast %input : tensor<4xi32> to tensor<?xi32>
  %1 = tensor.cast %0 : tensor<?xi32> to tensor<4xi32>
  // CHECK-NEXT: return %[[IN]]
  return %1 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_keep
// CHECK-SAME: %[[IN:.*]]: tensor<?x?xi32>
func @tensor.cast_chain_keep(%input: tensor<?x?xi32>) -> tensor<?x8xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor.cast %[[IN]]
  %0 = tensor.cast %input : tensor<?x?xi32> to tensor<4x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor.cast %[[C1]]
  %1 = tensor.cast %0 : tensor<4x?xi32> to tensor<?x8xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<?x8xi32>
}

// -----

// CHECK-LABEL: @tensor.cast_chain_invalid
// CHECK-SAME: %[[IN:.*]]: tensor<4x8xi32>
func @tensor.cast_chain_invalid(%input: tensor<4x8xi32>) -> tensor<8x4xi32> {
  // CHECK-NEXT: %[[C1:.*]] = tensor.cast %[[IN]]
  %0 = tensor.cast %input : tensor<4x8xi32> to tensor<?x?xi32>
  // CHECK-NEXT: %[[C2:.*]] = tensor.cast %[[C1]]
  %1 = tensor.cast %0 : tensor<?x?xi32> to tensor<8x4xi32>
  // CHECK-NEXT: return %[[C2]]
  return %1 : tensor<8x4xi32>
}

// -----

// CHECK-LABEL: func @fold_extract
func @fold_extract(%arg0 : index) -> (f32, f16, f16, i32) {
  %const_0 = constant 0 : index
  %const_1 = constant 1 : index
  %const_3 = constant 3 : index

  // Fold an extract into a splat.
  // CHECK-NEXT: [[C4:%.+]] = constant 4.{{0*}}e+00 : f32
  %0 = constant dense<4.0> : tensor<4xf32>
  %ext_1 = tensor.extract %0[%arg0] : tensor<4xf32>

  // Fold an extract into a sparse with a sparse index.
  // CHECK-NEXT: [[CM2:%.+]] = constant -2.{{0*}}e+00 : f16
  %1 = constant sparse<[[0, 0, 0], [1, 1, 1]],  [-5.0, -2.0]> : tensor<4x4x4xf16>
  %ext_2 = tensor.extract %1[%const_1, %const_1, %const_1] : tensor<4x4x4xf16>

  // Fold an extract into a sparse with a non sparse index.
  // CHECK-NEXT: [[C0:%.+]] = constant 0.{{0*}}e+00 : f16
  %2 = constant sparse<[[1, 1, 1]],  [-2.0]> : tensor<1x1x1xf16>
  %ext_3 = tensor.extract %2[%const_0, %const_0, %const_0] : tensor<1x1x1xf16>

  // Fold an extract into a dense tensor.
  // CHECK-NEXT: [[C64:%.+]] = constant 64 : i32
  %3 = constant dense<[[[1, -2, 1, 36]], [[0, 2, -1, 64]]]> : tensor<2x1x4xi32>
  %ext_4 = tensor.extract %3[%const_1, %const_0, %const_3] : tensor<2x1x4xi32>

  // CHECK-NEXT: return [[C4]], [[CM2]], [[C0]], [[C64]]
  return %ext_1, %ext_2, %ext_3, %ext_4 : f32, f16, f16, i32
}

// -----

// CHECK-LABEL: func @extract_from_tensor.cast
// CHECK-SAME: %[[TENSOR:.*]]: tensor<*xf32>
func @extract_from_tensor.cast(%tensor: tensor<*xf32>) -> f32 {
  // CHECK-NEXT: %[[C0:.*]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK-NOT: tensor.cast
  %casted = tensor.cast %tensor : tensor<*xf32> to tensor<?xf32>
  // CHECK-NEXT: tensor.extract %[[TENSOR]][%[[C0]]]
  %result = tensor.extract %casted[%c0] : tensor<?xf32>
  return %result : f32
}
