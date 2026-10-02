// RUN: mlir-opt %s --verify-each | FileCheck %s

// Allow i64 accumulators to facilitate incremental lowerings to/from TOSA.
// CHECK-LABEL: func.func @conv2d_i64
// CHECK: acc_type(i64)
func.func @conv2d_i64(%input: tensor<1x1x1x1xi16>, %weight: tensor<1x1x1x1xi8>, %bias: tensor<1xi64>, %izp: tensor<1xi16>, %wzp: tensor<1xi8>) -> tensor<1x1x1x1xi64> {
  %0 = tosa.conv2d %input, %weight, %bias, %izp, %wzp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i64) : (tensor<1x1x1x1xi16>, tensor<1x1x1x1xi8>, tensor<1xi64>, tensor<1xi16>, tensor<1xi8>) -> tensor<1x1x1x1xi64>
  return %0 : tensor<1x1x1x1xi64>
}
