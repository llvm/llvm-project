// RUN: mlir-opt %s -split-input-file --sparse-reinterpret-map | FileCheck %s

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

// -----

#BSR = #sparse_tensor.encoding<{
   map = ( i, j ) ->
      ( i floordiv 2 : dense,
        j floordiv 2 : compressed,
        i mod 2      : dense,
        j mod 2      : dense
      )
}>

// CHECK-LABEL:   func.func @sparse_foreach_reinterpret_map(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x4xf64
// CHECK:           %[[VAL_1:.*]] = bufferization.alloc_tensor() : tensor<1x2x2x2xf64
// CHECK:           %[[VAL_2:.*]] = sparse_tensor.reinterpret_map %[[VAL_0]] : tensor<2x4xf64
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.foreach in %[[VAL_2]] init(%[[VAL_1]])
// CHECK:           ^bb0(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index, %[[VAL_8:.*]]: index, %[[VAL_9:.*]]: f64, %[[VAL_10:.*]]: tensor<1x2x2x2xf64
// CHECK:             %[[VAL_11:.*]] = sparse_tensor.insert %[[VAL_9]] into %[[VAL_10]]{{\[}}%[[VAL_5]], %[[VAL_6]], %[[VAL_7]], %[[VAL_8]]] : tensor<1x2x2x2xf64
// CHECK:             sparse_tensor.yield %[[VAL_11]] : tensor<1x2x2x2xf64
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.reinterpret_map %[[VAL_4]] : tensor<1x2x2x2xf64
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.load %[[VAL_12]] hasInserts : tensor<2x4xf64
// CHECK:           return %[[VAL_13]] : tensor<2x4xf64
// CHECK:         }
func.func @sparse_foreach_reinterpret_map(%6 : tensor<2x4xf64, #BSR>) -> tensor<2x4xf64, #BSR> {
  %7 = bufferization.alloc_tensor() : tensor<2x4xf64, #BSR>
  %8 = sparse_tensor.foreach in %6 init(%7) : tensor<2x4xf64, #BSR>, tensor<2x4xf64, #BSR> -> tensor<2x4xf64, #BSR> do {
    ^bb0(%arg0: index, %arg1: index, %arg2: f64, %arg3: tensor<2x4xf64, #BSR>):
      %inserted = tensor.insert %arg2 into %arg3[%arg0, %arg1] : tensor<2x4xf64, #BSR>
      sparse_tensor.yield %inserted : tensor<2x4xf64, #BSR>
  }
  %9 = sparse_tensor.load %8 hasInserts : tensor<2x4xf64, #BSR>
  return %9 : tensor<2x4xf64, #BSR>
}
