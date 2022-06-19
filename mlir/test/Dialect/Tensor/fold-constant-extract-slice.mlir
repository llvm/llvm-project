// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-fold-constant-extract-slice %s | FileCheck %s

// CHECK-LABEL: func @slice_constant
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[CONST:.+]] = arith.constant dense<1.000000e+01> : tensor<1x1xf32>
//       CHECK:   return %[[CONST]] :  tensor<1x1xf32>
func.func @slice_constant(%arg0 : tensor<2x1xf32>) -> tensor<1x1xf32>
{
  %cst = arith.constant dense<[[10.0], [11.0]]> : tensor<2x1xf32>
  %slice = tensor.extract_slice %cst[0, 0] [1, 1] [1, 1] : tensor<2x1xf32> to tensor<1x1xf32>
  return %slice : tensor<1x1xf32>
}

// -----

// CHECK-LABEL: func @slice_constant_3x4
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[CONST:.+]] = arith.constant dense<{{\[}}[1.000000e+01, 9.000000e+00], [1.100000e+01, 1.200000e+01]]> : tensor<2x2xf32>
//       CHECK:   return %[[CONST]] :  tensor<2x2xf32>
func.func @slice_constant_3x4(%arg0 : tensor<3x4xf32>) -> tensor<2x2xf32>
{
  %cst = arith.constant dense<[[10.0, 9.0, 8.0, 7.0], [11.0, 12.0, 13.0, 14.0], [1.0, 3.0, 5.0, 7.0]]> : tensor<3x4xf32>
  %slice = tensor.extract_slice %cst[0, 0] [2, 2] [1, 1] : tensor<3x4xf32> to tensor<2x2xf32>
  return %slice : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @slice_constant_3x4_offsets
//   CHECK-NOT:   tensor.extract_slice
//       CHECK:   %[[CONST:.+]] = arith.constant dense<{{\[}}[1.200000e+01, 1.300000e+01], [3.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>
//       CHECK:   return %[[CONST]] :  tensor<2x2xf32>
func.func @slice_constant_3x4_offsets(%arg0 : tensor<3x4xf32>) -> tensor<2x2xf32>
{
  %cst = arith.constant dense<[[10.0, 9.0, 8.0, 7.0], [11.0, 12.0, 13.0, 14.0], [1.0, 3.0, 5.0, 7.0]]> : tensor<3x4xf32>
  %slice = tensor.extract_slice %cst[1, 1] [2, 2] [1, 1] : tensor<3x4xf32> to tensor<2x2xf32>
  return %slice : tensor<2x2xf32>
}

