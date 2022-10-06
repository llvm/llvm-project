// RUN: mlir-opt %s -sparsification -cse | FileCheck %s

#Dense = #sparse_tensor.encoding<{
  dimLevelType = [ "dense" , "dense" ]
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2.0"
}

// CHECK-LABEL: func.func @sparse_scale(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<1x1xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>)
// CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:         %[[VAL_3:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<1x1xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>> to memref<?xf32>
// CHECK:         %[[VAL_4:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:         %[[VAL_5:.*]] = arith.mulf %[[VAL_4]], %[[VAL_2]] : f32
// CHECK:         memref.store %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_1]]] : memref<?xf32>
// CHECK:         %[[VAL_6:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<1x1xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
// CHECK:         return %[[VAL_6]] : tensor<1x1xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense" ] }>>
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
