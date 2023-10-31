// RUN: mlir-opt %s -split-input-file  --sparse-reinterpret-map | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK-LABEL: func @sparse_nop(
//  CHECK-SAME: %[[A0:.*]]: tensor<?xf64, #sparse_tensor.encoding<{{{.*}}}>>)
//       CHECK: return %[[A0]]
func.func @sparse_nop(%arg0: tensor<?xf64, #SparseVector>) -> tensor<?xf64, #SparseVector> {
  return %arg0 : tensor<?xf64, #SparseVector>
}

// -----

#trait_mul = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A (in)
    affine_map<(i,j) -> (j,i)>,  // B (in, transposed)
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) *= A(i,j) * B(j,i)"
}

#BSR = #sparse_tensor.encoding<{   // 2x4 blocks
  map = (i, j) ->
    ( i floordiv 2 : dense
    , j floordiv 4 : compressed
    , i mod 2 : dense
    , j mod 4 : dense
    )
}>

// CHECK: #[[$map0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 4 + d3)>
// CHECK: #[[$map1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1 * 4 + d3, d0 * 2 + d2)>
// CHECK: #[[$map2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @mul(
// CHECK-SAME:  %[[A0:.*0]]: tensor<32x32xf32>,
// CHECK-SAME:  %[[A1:.*1]]: tensor<32x32xf32>,
// CHECK-SAME:  %[[A2:.*2]]: tensor<32x32xf32, #sparse_tensor.encoding<{{{.*}}}>>)
// CHECK:       %[[T0:.*]] = sparse_tensor.reinterpret_map %[[A2]]
// CHECK:       %[[T1:.*]] = linalg.generic {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// CHECK:       %[[T2:.*]] = sparse_tensor.reinterpret_map %[[T1]]
// CHECK:       return %[[T2]] : tensor<32x32xf32, #sparse_tensor.encoding<{{{.*}}}>>
func.func @mul(%arg0: tensor<32x32xf32>,
               %arg1: tensor<32x32xf32>,
               %arg2: tensor<32x32xf32, #BSR>) -> tensor<32x32xf32, #BSR> {
  %0 = linalg.generic #trait_mul
    ins(%arg0, %arg1: tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg2: tensor<32x32xf32, #BSR>) {
      ^bb(%x: f32, %y : f32, %z : f32):
        %1 = arith.mulf %x, %y : f32
        %2 = arith.mulf %1, %z : f32
        linalg.yield %2 : f32
  } -> tensor<32x32xf32, #BSR>
  return %0 : tensor<32x32xf32, #BSR>
}

