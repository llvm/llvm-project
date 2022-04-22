// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize=test-analysis-only -split-input-file | FileCheck %s

/// All combinations of matmul(fill(extract(init_tensor)), fill(extract(%init_tensor)), %arg2)
/// These should all be inplaceable except the first op.

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1234(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1243(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1324(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1342(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1423(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_1432(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2134(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2143(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2314(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2341(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2413(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_2431(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["none", "false"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3124(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3142(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3214(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3241(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %2[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %4 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3412(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_3421(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4123(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4132(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4213(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256xf32>) -> tensor<256x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %1[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%3, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4231(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4312(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}

// -----

// CHECK-LABEL: func @fill_extract_matmul_
func.func @fill_extract_matmul_4321(
    %arg0: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg1: tensor<518x518xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false},
    %arg2: tensor<256x256xf32> {bufferization.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true})
  -> tensor<256x256xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = linalg.init_tensor [256, 256] : tensor<256x256xf32>

  // CHECK: {__inplace_operands_attr__ = ["false"]}
  %4 = tensor.extract_slice %0[0, 0] [16, 256] [1, 1] : tensor<256x256xf32> to tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["true"]}
  %3 = tensor.extract_slice %0[0, 0] [256, 16] [1, 1] : tensor<256x256xf32> to tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %2 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<16x256xf32>) -> tensor<16x256xf32>
  // CHECK: {__inplace_operands_attr__ = ["none", "true"]}
  %1 = linalg.fill ins(%cst : f32) outs(%3 : tensor<256x16xf32>) -> tensor<256x16xf32>
  // CHECK: {__inplace_operands_attr__ = ["true", "true", "true"]}
  %5 = linalg.matmul ins(%1, %2 : tensor<256x16xf32>, tensor<16x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %5 : tensor<256x256xf32>
}
