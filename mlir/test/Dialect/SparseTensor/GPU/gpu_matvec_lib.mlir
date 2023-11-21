// RUN: mlir-opt %s --linalg-generalize-named-ops --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

module {

// CHECK-LABEL:   func.func @matvec(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<?xf64>) -> tensor<?xf64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = gpu.wait async
// CHECK:           %[[VAL_12:.*]] = memref.dim %[[VAL_8]], %[[VAL_3]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = gpu.alloc async {{\[}}%[[VAL_11]]] (%[[VAL_12]]) : memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = gpu.memcpy async {{\[}}%[[VAL_14]]] %[[VAL_13]], %[[VAL_8]] : memref<?xindex>, memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_16:.*]] = gpu.wait async
// CHECK:           %[[VAL_17:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_18:.*]], %[[VAL_19:.*]] = gpu.alloc async {{\[}}%[[VAL_16]]] (%[[VAL_17]]) : memref<?xindex>
// CHECK:           %[[VAL_20:.*]] = gpu.memcpy async {{\[}}%[[VAL_19]]] %[[VAL_18]], %[[VAL_9]] : memref<?xindex>, memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_21:.*]] = gpu.wait async
// CHECK:           %[[VAL_22:.*]] = memref.dim %[[VAL_10]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = gpu.alloc async {{\[}}%[[VAL_21]]] (%[[VAL_22]]) : memref<?xf64>
// CHECK:           %[[VAL_25:.*]] = gpu.memcpy async {{\[}}%[[VAL_24]]] %[[VAL_23]], %[[VAL_10]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_26:.*]] = bufferization.to_memref %[[VAL_1]] : memref<?xf64>
// CHECK:           %[[VAL_27:.*]] = gpu.wait async
// CHECK:           %[[VAL_28:.*]] = memref.dim %[[VAL_26]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_29:.*]], %[[VAL_30:.*]] = gpu.alloc async {{\[}}%[[VAL_27]]] (%[[VAL_28]]) : memref<?xf64>
// CHECK:           %[[VAL_31:.*]] = gpu.memcpy async {{\[}}%[[VAL_30]]] %[[VAL_29]], %[[VAL_26]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_32:.*]] = bufferization.to_memref %[[VAL_2]] : memref<?xf64>
// CHECK:           %[[VAL_33:.*]] = gpu.wait async
// CHECK:           %[[VAL_34:.*]] = memref.dim %[[VAL_32]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = gpu.alloc async {{\[}}%[[VAL_33]]] (%[[VAL_34]]) : memref<?xf64>
// CHECK:           %[[VAL_37:.*]] = gpu.memcpy async {{\[}}%[[VAL_36]]] %[[VAL_35]], %[[VAL_32]] : memref<?xf64>, memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_15]], %[[VAL_20]], %[[VAL_25]], %[[VAL_31]], %[[VAL_37]]]
// CHECK:           %[[VAL_38:.*]] = gpu.wait async
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = gpu.create_coo async {{\[}}%[[VAL_38]]] %[[VAL_6]], %[[VAL_7]], %[[VAL_5]], %[[VAL_13]], %[[VAL_18]], %[[VAL_23]] : memref<?xindex>, memref<?xindex>, memref<?xf64>
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_42]]] %[[VAL_29]], %[[VAL_7]] : index into memref<?xf64>
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_44]]] %[[VAL_35]], %[[VAL_6]] : index into memref<?xf64>
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = gpu.spmv_buffer_size async {{\[}}%[[VAL_46]]] %[[VAL_41]], %[[VAL_43]], %[[VAL_45]]
// CHECK:           %[[VAL_49:.*]], %[[VAL_50:.*]] = gpu.alloc async {{\[}}%[[VAL_48]]] (%[[VAL_47]]) : memref<?xi8>
// CHECK:           %[[VAL_51:.*]] = gpu.spmv async {{\[}}%[[VAL_50]]] %[[VAL_41]], %[[VAL_43]], %[[VAL_45]], %[[VAL_49]] : memref<?xi8>
// CHECK:           %[[VAL_52:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_51]]] %[[VAL_41]]
// CHECK:           %[[VAL_53:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_52]]] %[[VAL_43]]
// CHECK:           %[[VAL_54:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_53]]] %[[VAL_45]]
// CHECK:           %[[VAL_56:.*]] = gpu.dealloc async {{\[}}%[[VAL_54]]] %[[VAL_13]] : memref<?xindex>
// CHECK:           %[[VAL_57:.*]] = gpu.dealloc async {{\[}}%[[VAL_56]]] %[[VAL_18]] : memref<?xindex>
// CHECK:           %[[VAL_58:.*]] = gpu.dealloc async {{\[}}%[[VAL_57]]] %[[VAL_23]] : memref<?xf64>
// CHECK:           %[[VAL_59:.*]] = gpu.dealloc async {{\[}}%[[VAL_58]]] %[[VAL_49]] : memref<?xi8>
// CHECK:           %[[VAL_60:.*]] = gpu.dealloc async {{\[}}%[[VAL_59]]] %[[VAL_29]] : memref<?xf64>
// CHECK:           %[[VAL_61:.*]] = gpu.memcpy async {{\[}}%[[VAL_60]]] %[[VAL_32]], %[[VAL_35]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_62:.*]] = gpu.dealloc async {{\[}}%[[VAL_61]]] %[[VAL_35]] : memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_62]]]
// CHECK:           %[[VAL_63:.*]] = bufferization.to_tensor %[[VAL_32]] : memref<?xf64>
// CHECK:           return %[[VAL_63]] : tensor<?xf64>
// CHECK:         }
func.func @matvec(%A: tensor<?x?xf64, #SortedCOO>,
                  %x: tensor<?xf64>,
                  %y_in: tensor<?xf64>) -> tensor<?xf64> {
  %y_out = linalg.matvec
    ins(%A, %x: tensor<?x?xf64, #SortedCOO>, tensor<?xf64>)
    outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
  return %y_out : tensor<?xf64>
}

}
