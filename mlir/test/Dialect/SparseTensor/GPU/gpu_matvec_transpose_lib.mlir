// RUN: mlir-opt %s --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

module {

// Check y += transpose(A) * x. The sparse descriptor keeps A's physical
// dimensions, while the input and output vector lengths are rows(A) and cols(A).

// CHECK-LABEL: func.func @matvec_transpose(
// CHECK-SAME:      %[[A:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[X:.*]]: tensor<?xf64>,
// CHECK-SAME:      %[[Y:.*]]: tensor<?xf64>) -> tensor<?xf64> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
// CHECK-DAG:     %[[ROWS:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[COLS:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG:     %[[X_HOST:.*]] = bufferization.to_buffer %[[X]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[X_DEVICE:.*]], %[[X_HOST]] :
// CHECK-DAG:     %[[Y_HOST:.*]] = bufferization.to_buffer %[[Y]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[Y_DEVICE:.*]], %[[Y_HOST]] :
// CHECK:         %[[SPMAT:.*]], {{.*}} = gpu.create_coo async {{\[.*\]}} %[[ROWS]], %[[COLS]], %[[NNZ]],
// CHECK-DAG:     %[[DNX:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[X_DEVICE]], %[[ROWS]] : index into memref<?xf64>
// CHECK-DAG:     %[[DNY:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[Y_DEVICE]], %[[COLS]] : index into memref<?xf64>
// CHECK:         gpu.spmv_buffer_size async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNX]], %[[DNY]] into f64
// CHECK:         gpu.spmv async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNX]], %[[DNY]],
// CHECK:         return
// CHECK:       }
func.func @matvec_transpose(%A: tensor<?x?xf64, #SortedCOO>,
                            %x: tensor<?xf64>,
                            %y_in: tensor<?xf64>) -> tensor<?xf64> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (j, i)>,
      affine_map<(i, j) -> (j)>,
      affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%A, %x : tensor<?x?xf64, #SortedCOO>, tensor<?xf64>)
  outs(%y_in : tensor<?xf64>) {
  ^bb0(%a: f64, %xval: f64, %yval: f64):
    %product = arith.mulf %a, %xval : f64
    %sum = arith.addf %yval, %product : f64
    linalg.yield %sum : f64
  } -> tensor<?xf64>
  return %result : tensor<?xf64>
}

}
