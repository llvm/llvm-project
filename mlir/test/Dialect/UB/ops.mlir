// RUN: mlir-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @poison()
//       CHECK:   %{{.*}} = ub.poison : i32
func.func @poison() -> i32 {
  %0 = ub.poison : i32
  return %0 : i32
}

// CHECK-LABEL: func @poison_full_form()
//       CHECK:   %{{.*}} = ub.poison : i32
func.func @poison_full_form() -> i32 {
  %0 = ub.poison <#ub.poison> : i32
  return %0 : i32
}

// CHECK-LABEL: func @poison_complex()
//       CHECK:   %{{.*}} = ub.poison : complex<f32>
func.func @poison_complex() -> complex<f32> {
  %0 = ub.poison : complex<f32>
  return %0 : complex<f32>
}

// CHECK-LABEL: func @poison_vec()
//       CHECK:   %{{.*}} = ub.poison : vector<4xi64>
func.func @poison_vec() -> vector<4xi64> {
  %0 = ub.poison : vector<4xi64>
  return %0 : vector<4xi64>
}

// CHECK-LABEL: func @poison_tensor()
//       CHECK:   %{{.*}} = ub.poison : tensor<8x?xf64>
func.func @poison_tensor() -> tensor<8x?xf64> {
  %0 = ub.poison : tensor<8x?xf64>
  return %0 : tensor<8x?xf64>
}
