// RUN: mlir-opt %s -canonicalize | FileCheck %s

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
