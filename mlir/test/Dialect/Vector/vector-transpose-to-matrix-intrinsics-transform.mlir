// RUN: mlir-opt %s --convert-vector-to-llvm='vector-transpose-lowering=flat' --split-input-file | FileCheck %s

// CHECK-LABEL: func @transpose(
func.func @transpose(%arg0: vector<2x4xf32>) -> vector<4x2xf32> {
  // CHECK:       llvm.intr.matrix.transpose %{{.*}} {columns = 2 : i32, rows = 4 : i32} : vector<8xf32> into vector<8xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<2x4xf32> to vector<4x2xf32>
  return %0 : vector<4x2xf32>
}

/// Scalable vectors are not supported

// CHECK-LABEL: func @transpose_scalable(
func.func @transpose_scalable(%arg0: vector<2x[4]xf32>) -> vector<[4]x2xf32> {
  // CHECK-NOT:       llvm.intr.matrix.transpose
  // CHECK:           vector.transpose
  %0 = vector.transpose %arg0, [1, 0] : vector<2x[4]xf32> to vector<[4]x2xf32>
  return %0 : vector<[4]x2xf32>
}
