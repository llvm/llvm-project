// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=2,4,8 vectorize vectorize-contraction-to=matrixintrinsics unroll-vector-transfers=true" -split-input-file | FileCheck %s --check-prefix=CHECK-INTRINSIC
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 promote promote-full-tile-pad register-tile-sizes=2,4,8 vectorize vectorize-contraction-to=outerproduct split-transfers=true unroll-vector-transfers=false" -split-input-file | FileCheck %s --check-prefix=CHECK-OUTER
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 tile-interchange=1,2,0 generalize iterator-interchange=0,2,1" -split-input-file | FileCheck %s --check-prefix=CHECK-INTERCHANGE
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 pad pack-paddings=1,1,0 hoist-paddings=3,3,0" -split-input-file | FileCheck %s --check-prefix=CHECK-PAD

// CHECK-INTRINSIC: func @matmul(
//     CHECK-OUTER: func @matmul(
func @matmul(%arg0: memref<72x72xf32>, %arg1: memref<72x72xf32>, %arg2: memref<72x72xf32>) {

  // Check the matrix intrinsic lowering is triggered.
  //      CHECK-INTRINSIC: vector.matrix_multiply
  // CHECK-INTRINSIC-SAME: {lhs_columns = 8 : i32, lhs_rows = 2 : i32, rhs_columns = 4 : i32}
  // CHECK-INTRINSIC-SAME: (vector<16xf32>, vector<32xf32>) -> vector<8xf32>

  // Check the outer product lowering is triggered.
  //          CHECK-OUTER: vector.outerproduct {{.*}} : vector<2xf32>, vector<4xf32>
  linalg.matmul ins(%arg0, %arg1: memref<72x72xf32>, memref<72x72xf32>) outs(%arg2: memref<72x72xf32>)
  return
}

// -----

// CHECK-INTERCHANGE: func @matmul(
func @matmul(%arg0: tensor<72x72xf32>, %arg1: tensor<72x72xf32>, %arg2: tensor<72x72xf32>) -> tensor<72x72xf32> {
  //  CHECK-INTERCHANGE-DAG: %[[C16:.*]] = arith.constant 16
  //  CHECK-INTERCHANGE-DAG: %[[C32:.*]] = arith.constant 32
  //  CHECK-INTERCHANGE-DAG: %[[C64:.*]] = arith.constant 64

  // Check the tile loops are interchanged.
  //      CHECK-INTERCHANGE: scf.for {{.*}} step %[[C32]]
  //      CHECK-INTERCHANGE:   scf.for {{.*}} step %[[C64]]
  //      CHECK-INTERCHANGE:    scf.for {{.*}} step %[[C16]]

  // Check the operation has been generalized and interchanged.
  //      CHECK-INTERCHANGE:      linalg.generic
  // CHECK-INTERCHANGE-SAME:      iterator_types = ["parallel", "reduction", "parallel"]
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<72x72xf32>, tensor<72x72xf32>) outs(%arg2: tensor<72x72xf32>) -> tensor<72x72xf32>
  return %0 : tensor<72x72xf32>
}

// -----

//         CHECK-PAD: func @matmul(
func @matmul(%arg0: tensor<72x72xf32>, %arg1: tensor<72x72xf32>, %arg2: tensor<72x72xf32>) -> tensor<72x72xf32> {

  // Check the padding of the input operands has been hoisted out of the tile loop nest.
  //      CHECK-PAD-COUNT=2: linalg.pad_tensor %{{.*}} nofold
  //      CHECK-PAD-COUNT=3: scf.for
  //              CHECK-PAD: linalg.matmul
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<72x72xf32>, tensor<72x72xf32>) outs(%arg2: tensor<72x72xf32>) -> tensor<72x72xf32>
  return %0 : tensor<72x72xf32>
}

