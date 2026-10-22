// RUN: mlir-opt %s --linalg-generalize-named-ops --sparse-gpu-codegen="num-threads=0" | FileCheck %s --enable-var-scope

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

// CHECK-LABEL: func.func @matmul_transpose_a(
// CHECK-SAME:      %[[A:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[B:.*]]: tensor<?x?xf64>,
// CHECK-SAME:      %[[C:.*]]: tensor<?x?xf64>) -> tensor<?x?xf64> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
// CHECK-DAG:     %[[ROWS:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[COLS:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG:     %[[N:.*]] = tensor.dim %[[B]], %[[C1]]
// CHECK-DAG:     %[[B_HOST:.*]] = bufferization.to_buffer %[[B]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[B_DEVICE:.*]], %[[B_HOST]] :
// CHECK-DAG:     %[[C_HOST:.*]] = bufferization.to_buffer %[[C]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[C_DEVICE:.*]], %[[C_HOST]] :
// CHECK:         %[[SPMAT:.*]], {{.*}} = gpu.create_coo async {{\[.*\]}} %[[ROWS]], %[[COLS]], %[[NNZ]],
// CHECK:         %[[DNB:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[B_DEVICE]], %[[ROWS]], %[[N]] : index, index into memref<?x?xf64>
// CHECK:         %[[DNC:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[C_DEVICE]], %[[COLS]], %[[N]] : index, index into memref<?x?xf64>
// CHECK:         gpu.spmm_buffer_size async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNB]], %[[DNC]] : index into f64
// CHECK:         gpu.spmm async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNB]], %[[DNC]],
// CHECK:         return
// CHECK:       }
func.func @matmul_transpose_a(%A: tensor<?x?xf64, #SortedCOO>,
                              %B: tensor<?x?xf64>,
                              %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
    indexing_maps = [
      affine_map<(i, j, k) -> (k, i)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    ins(%A, %B: tensor<?x?xf64, #SortedCOO>, tensor<?x?xf64>)
    outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}

// CHECK-LABEL: func.func @matmul_transpose_b(
// CHECK-SAME:      %[[A:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[B:.*]]: tensor<?x?xf64>,
// CHECK-SAME:      %[[C:.*]]: tensor<?x?xf64>) -> tensor<?x?xf64> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
// CHECK-DAG:     %[[ROWS:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[COLS:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG:     %[[N:.*]] = tensor.dim %[[B]], %[[C0]]
// CHECK-DAG:     %[[B_HOST:.*]] = bufferization.to_buffer %[[B]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[B_DEVICE:.*]], %[[B_HOST]] :
// CHECK-DAG:     %[[C_HOST:.*]] = bufferization.to_buffer %[[C]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[C_DEVICE:.*]], %[[C_HOST]] :
// CHECK:         %[[SPMAT:.*]], {{.*}} = gpu.create_coo async {{\[.*\]}} %[[ROWS]], %[[COLS]], %[[NNZ]],
// CHECK:         %[[DNB:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[B_DEVICE]], %[[N]], %[[COLS]] : index, index into memref<?x?xf64>
// CHECK:         %[[DNC:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[C_DEVICE]], %[[ROWS]], %[[N]] : index, index into memref<?x?xf64>
// CHECK:         gpu.spmm_buffer_size async {{\[.*\]}} %[[SPMAT]], %[[DNB]]{TRANSPOSE}, %[[DNC]] : index into f64
// CHECK:         gpu.spmm async {{\[.*\]}} %[[SPMAT]], %[[DNB]]{TRANSPOSE}, %[[DNC]],
// CHECK:         return
// CHECK:       }
func.func @matmul_transpose_b(%A: tensor<?x?xf64, #SortedCOO>,
                              %B: tensor<?x?xf64>,
                              %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (j, k)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    ins(%A, %B: tensor<?x?xf64, #SortedCOO>, tensor<?x?xf64>)
    outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}

// CHECK-LABEL: func.func @matmul_transpose_ab(
// CHECK-SAME:      %[[A:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[B:.*]]: tensor<?x?xf64>,
// CHECK-SAME:      %[[C:.*]]: tensor<?x?xf64>) -> tensor<?x?xf64> {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[NNZ:.*]] = sparse_tensor.number_of_entries %[[A]]
// CHECK-DAG:     %[[ROWS:.*]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG:     %[[COLS:.*]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG:     %[[N:.*]] = tensor.dim %[[B]], %[[C0]]
// CHECK-DAG:     %[[B_HOST:.*]] = bufferization.to_buffer %[[B]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[B_DEVICE:.*]], %[[B_HOST]] :
// CHECK-DAG:     %[[C_HOST:.*]] = bufferization.to_buffer %[[C]]
// CHECK-DAG:     gpu.memcpy async {{\[.*\]}} %[[C_DEVICE:.*]], %[[C_HOST]] :
// CHECK:         %[[SPMAT:.*]], {{.*}} = gpu.create_coo async {{\[.*\]}} %[[ROWS]], %[[COLS]], %[[NNZ]],
// CHECK:         %[[DNB:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[B_DEVICE]], %[[N]], %[[ROWS]] : index, index into memref<?x?xf64>
// CHECK:         %[[DNC:.*]], {{.*}} = gpu.create_dn_tensor async {{\[.*\]}} %[[C_DEVICE]], %[[COLS]], %[[N]] : index, index into memref<?x?xf64>
// CHECK:         gpu.spmm_buffer_size async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNB]]{TRANSPOSE}, %[[DNC]] : index into f64
// CHECK:         gpu.spmm async {{\[.*\]}} %[[SPMAT]]{TRANSPOSE}, %[[DNB]]{TRANSPOSE}, %[[DNC]],
// CHECK:         return
// CHECK:       }
func.func @matmul_transpose_ab(%A: tensor<?x?xf64, #SortedCOO>,
                               %B: tensor<?x?xf64>,
                               %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
    indexing_maps = [
      affine_map<(i, j, k) -> (k, i)>,
      affine_map<(i, j, k) -> (j, k)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    ins(%A, %B: tensor<?x?xf64, #SortedCOO>, tensor<?x?xf64>)
    outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func @spgemm_transpose_a(
// CHECK-NOT:     gpu.
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.
// CHECK:         return
func.func @spgemm_transpose_a(%A: tensor<?x?xf64, #CSR>,
                              %B: tensor<?x?xf64, #CSR>,
                              %C_in: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR> {
  %C_out = linalg.matmul
    indexing_maps = [
      affine_map<(i, j, k) -> (k, i)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    ins(%A, %B: tensor<?x?xf64, #CSR>, tensor<?x?xf64, #CSR>)
    outs(%C_in: tensor<?x?xf64, #CSR>) -> tensor<?x?xf64, #CSR>
  return %C_out : tensor<?x?xf64, #CSR>
}

#NV_24 = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j floordiv 4 : dense, j mod 4 : structured[2, 4])
}>

// CHECK-LABEL: func.func @matmul_2to4_transpose_b(
// CHECK-NOT:     gpu.
// CHECK:         linalg.generic
// CHECK-NOT:     gpu.
// CHECK:         return
func.func @matmul_2to4_transpose_b(%A: tensor<?x?xf16>,
                                   %B: tensor<?x?xf16>,
                                   %C_in: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %A_sparse = sparse_tensor.convert %A : tensor<?x?xf16> to tensor<?x?xf16, #NV_24>
  %C_out = linalg.matmul
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (j, k)>,
      affine_map<(i, j, k) -> (i, j)>
    ]
    ins(%A_sparse, %B: tensor<?x?xf16, #NV_24>, tensor<?x?xf16>)
    outs(%C_in: tensor<?x?xf16>) -> tensor<?x?xf16>
  return %C_out : tensor<?x?xf16>
}
