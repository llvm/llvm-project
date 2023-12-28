// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse | FileCheck %s

#Dense = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : dense)
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2.0"
}

// CHECK-LABEL:   func.func @sparse_scale(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1xf32, #sparse{{[0-9]*}}>) -> tensor<1x1xf32, #sparse{{[0-9]*}}> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = sparse_tensor.insert %[[VAL_2]] into %[[VAL_0]]{{\[}}%[[VAL_1]], %[[VAL_1]]] : tensor<1x1xf32, #sparse{{[0-9]*}}>
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.load %[[VAL_3]] hasInserts : tensor<1x1xf32, #sparse{{[0-9]*}}>
// CHECK:           return %[[VAL_4]] : tensor<1x1xf32, #sparse{{[0-9]*}}>
// CHECK:         }
func.func @sparse_scale(%argx: tensor<1x1xf32, #Dense>) -> tensor<1x1xf32, #Dense> {
  %c = arith.constant 2.0 : f32
  %0 = linalg.generic #trait_scale
    outs(%argx: tensor<1x1xf32, #Dense>) {
      ^bb(%x: f32):
        %1 = arith.mulf %x, %c : f32
        linalg.yield %1 : f32
  } -> tensor<1x1xf32, #Dense>
  return %0 : tensor<1x1xf32, #Dense>
}
