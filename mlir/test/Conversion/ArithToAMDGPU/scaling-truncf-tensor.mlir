// RUN: mlir-opt %s -convert-arith-to-amdgpu=chipset=gfx950 | FileCheck %s

// CHECK-LABEL: func.func @m0
// CHECK: arith.scaling_truncf
func.func @m0(%arg0: tensor<16xf16>, %arg1: tensor<16xf8E8M0FNU>) -> tensor<16xf4E2M1FN> {
  %0 = arith.scaling_truncf %arg0, %arg1 : tensor<16xf16>, tensor<16xf8E8M0FNU> to tensor<16xf4E2M1FN>
  return %0 : tensor<16xf4E2M1FN>
}

// CHECK-LABEL: func.func @m1
// CHECK: arith.constant
// CHECK: arith.scaling_truncf
func.func @m1(%arg0: tensor<4xf32>) -> tensor<4xf4E2M1FN> {
  %cst = arith.constant dense<1.000000e+00> : tensor<4xf8E8M0FNU>
  %0 = arith.scaling_truncf %arg0, %cst : tensor<4xf32>, tensor<4xf8E8M0FNU> to tensor<4xf4E2M1FN>
  return %0 : tensor<4xf4E2M1FN>
}
