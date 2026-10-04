// RUN: mlir-opt %s --stage-sparse-ops | FileCheck %s

#SparseTensor = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d2 : dense, d0 : compressed, d1 : compressed)
}>

// CHECK-LABEL: func.func @sparse_convert_3d
// CHECK:         %[[TMP:.*]] = sparse_tensor.reorder_coo
// CHECK:         bufferization.dealloc_tensor %[[TMP]]
func.func @sparse_convert_3d(%arg0: tensor<?x?x?xf64>) -> tensor<?x?x?xf64, #SparseTensor> {
  %0 = sparse_tensor.convert %arg0 : tensor<?x?x?xf64> to tensor<?x?x?xf64, #SparseTensor>
  return %0 : tensor<?x?x?xf64, #SparseTensor>
}
