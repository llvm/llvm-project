// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=2,4,8 vectorize vectorize-contraction-to=matrixintrinsics unroll-vector-transfers=true" -split-input-file | FileCheck %s --check-prefix=CHECK-INTRINSIC
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 promote promote-full-tile-pad register-tile-sizes=2,4,8 vectorize vectorize-contraction-to=outerproduct split-transfers=true unroll-vector-transfers=false" -split-input-file | FileCheck %s --check-prefix=CHECK-OUTER
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 tile-interchange=1,2,0 generalize iterator-interchange=0,2,1" -split-input-file | FileCheck %s --check-prefix=CHECK-INTERCHANGE
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1,2 pack-paddings=1,1,0 hoist-paddings=3,3,0" -split-input-file | FileCheck %s --check-prefix=CHECK-PAD
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=matmul anchor-op=linalg.matmul tile-sizes=16,32,64 fuse pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1,2 vectorize" -split-input-file | FileCheck %s --check-prefix=CHECK-FUSE
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-func=conv anchor-op=linalg.conv_2d_nhwc_hwcf tile-sizes=1,1,8,32,1,1,8 fuse pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1,2 decompose vectorize vectorize-padding" -split-input-file | FileCheck %s --check-prefix=CHECK-DECOMP

// CHECK-INTRINSIC: func @matmul(
//     CHECK-OUTER: func @matmul(
func.func @matmul(%arg0: memref<72x72xf32>, %arg1: memref<72x72xf32>, %arg2: memref<72x72xf32>) {

  // Check the matrix intrinsic lowering is triggered.
  //      CHECK-INTRINSIC: vector.matrix_multiply
  // CHECK-INTRINSIC-SAME: {lhs_columns = 8 : i32, lhs_rows = 2 : i32, rhs_columns = 4 : i32}
  // CHECK-INTRINSIC-SAME: (vector<16xf32>, vector<32xf32>) -> vector<8xf32>

  // Check the outer product lowering is triggered.
  //          CHECK-OUTER: vector.outerproduct {{.*}} : vector<2xf32>, vector<4xf32>
  linalg.matmul ins(%arg0, %arg1: memref<72x72xf32>, memref<72x72xf32>) outs(%arg2: memref<72x72xf32>)
  func.return
}

// -----

// CHECK-INTERCHANGE: func @matmul(
func.func @matmul(%arg0: tensor<72x72xf32>, %arg1: tensor<72x72xf32>, %arg2: tensor<72x72xf32>) -> tensor<72x72xf32> {
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
  func.return %0 : tensor<72x72xf32>
}

// -----

//     CHECK-PAD-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0) -> (-d0 + 72, 16)>

//         CHECK-PAD: func @matmul(
func.func @matmul(%arg0: tensor<72x72xf32>, %arg1: tensor<72x72xf32>, %arg2: tensor<72x72xf32>) -> tensor<72x72xf32> {

  // Check the padding of the input operands has been hoisted out of the tile loop nest.
  //      CHECK-PAD-COUNT=2: tensor.pad %{{.*}} nofold
  //              CHECK-PAD: scf.for
  // Check CSE eliminates the duplicate min operations introduced by tiling.
  //              CHECK-PAD: affine.min #[[MAP0]]
  //          CHECK-PAD-NOT: affine.min #[[MAP0]]
  //      CHECK-PAD-COUNT=2: scf.for
  //              CHECK-PAD: linalg.matmul
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<72x72xf32>, tensor<72x72xf32>) outs(%arg2: tensor<72x72xf32>) -> tensor<72x72xf32>
  func.return %0 : tensor<72x72xf32>
}

// -----

//         CHECK-FUSE: func @matmul(
func.func @matmul(%arg0: tensor<72x72xf32>, %arg1: tensor<72x72xf32>, %arg2: tensor<72x72xf32>) -> tensor<72x72xf32> {

  // Check the padding and vectorization applies to the fill operation due to the empty anchor op string.
  //        CHECK-FUSE:  %[[CST:.*]] = arith.constant dense<0.000000e+00>
  //        CHECK-FUSE:  vector.transfer_write %[[CST]]
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<72x72xf32>) -> tensor<72x72xf32>

  // Check the matmul is padded and vectorized despite the empty anchor op string.
  //        CHECK-FUSE:  vector.outerproduct
  %1 = linalg.matmul ins(%arg0, %arg1: tensor<72x72xf32>, tensor<72x72xf32>) outs(%0: tensor<72x72xf32>) -> tensor<72x72xf32>
  func.return %1 : tensor<72x72xf32>
}

// -----

//         CHECK-DECOMP: func @conv(
func.func @conv(%arg0: tensor<8x18x17x32xf32>, %arg1: tensor<3x3x32x64xf32>, %arg2: tensor<8x16x15x64xf32>) -> tensor<8x16x15x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<8x16x15x64xf32>) -> tensor<8x16x15x64xf32>

  // Check the conv is padded by a rank-reducing vector transfer op pair.
  //        CHECK-DECOMP:  vector.transfer_read {{.*}}: tensor<1x1x?x8xf32>, vector<1x8x8xf32>
  //        CHECK-DECOMP:  vector.outerproduct
  //        CHECK-DECOMP:  vector.transfer_write {{.*}}: vector<1x8x32xf32>, tensor<1x1x?x32xf32>
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<8x18x17x32xf32>, tensor<3x3x32x64xf32>) outs(%0 : tensor<8x16x15x64xf32>) -> tensor<8x16x15x64xf32>
  func.return %1 : tensor<8x16x15x64xf32>
}
