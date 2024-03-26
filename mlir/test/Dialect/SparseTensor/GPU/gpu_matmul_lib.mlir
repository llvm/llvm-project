// RUN: mlir-opt %s --linalg-generalize-named-ops --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

//
// Compute matrix matrix C = AB
//
// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse{{[0-9]*}}>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<?x?xf64>) -> tensor<?x?xf64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse{{[0-9]*}}>
// CHECK-DAG:       %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// CHECK:           %[[VAL_12:.*]] = gpu.wait async
// CHECK:           %[[VAL_13:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]], %[[VAL_15:.*]] = gpu.alloc async {{\[}}%[[VAL_12]]] (%[[VAL_13]]) : memref<?xindex>
// CHECK:           %[[VAL_16:.*]] = gpu.memcpy async {{\[}}%[[VAL_15]]] %[[VAL_14]], %[[VAL_9]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = gpu.wait async
// CHECK:           %[[VAL_18:.*]] = memref.dim %[[VAL_10]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]], %[[VAL_20:.*]] = gpu.alloc async {{\[}}%[[VAL_17]]] (%[[VAL_18]]) : memref<?xindex>
// CHECK:           %[[VAL_21:.*]] = gpu.memcpy async {{\[}}%[[VAL_20]]] %[[VAL_19]], %[[VAL_10]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = gpu.wait async
// CHECK:           %[[VAL_23:.*]] = memref.dim %[[VAL_11]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = gpu.alloc async {{\[}}%[[VAL_22]]] (%[[VAL_23]]) : memref<?xf64>
// CHECK:           %[[VAL_26:.*]] = gpu.memcpy async {{\[}}%[[VAL_25]]] %[[VAL_24]], %[[VAL_11]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_27:.*]] = bufferization.to_memref %[[VAL_1]] : memref<?x?xf64>
// CHECK:           %[[VAL_28:.*]] = gpu.wait async
// CHECK:           %[[VAL_29:.*]] = memref.dim %[[VAL_27]], %[[VAL_3]] : memref<?x?xf64>
// CHECK:           %[[VAL_30:.*]] = memref.dim %[[VAL_27]], %[[VAL_4]] : memref<?x?xf64>
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = gpu.alloc async {{\[}}%[[VAL_28]]] (%[[VAL_29]], %[[VAL_30]]) : memref<?x?xf64>
// CHECK:           %[[VAL_33:.*]] = gpu.memcpy async {{\[}}%[[VAL_32]]] %[[VAL_31]], %[[VAL_27]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           %[[VAL_34:.*]] = bufferization.to_memref %[[VAL_2]] : memref<?x?xf64>
// CHECK:           %[[VAL_35:.*]] = gpu.wait async
// CHECK:           %[[VAL_36:.*]] = memref.dim %[[VAL_34]], %[[VAL_3]] : memref<?x?xf64>
// CHECK:           %[[VAL_37:.*]] = memref.dim %[[VAL_34]], %[[VAL_4]] : memref<?x?xf64>
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = gpu.alloc async {{\[}}%[[VAL_35]]] (%[[VAL_36]], %[[VAL_37]]) : memref<?x?xf64>
// CHECK:           %[[VAL_40:.*]] = gpu.memcpy async {{\[}}%[[VAL_39]]] %[[VAL_38]], %[[VAL_34]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_16]], %[[VAL_21]], %[[VAL_26]], %[[VAL_33]], %[[VAL_40]]]
// CHECK:           %[[VAL_41:.*]] = gpu.wait async
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = gpu.create_csr async {{\[}}%[[VAL_41]]] %[[VAL_6]], %[[VAL_7]], %[[VAL_5]], %[[VAL_14]], %[[VAL_19]], %[[VAL_24]] : memref<?xindex>, memref<?xindex>, memref<?xf64>
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_45]]] %[[VAL_31]], %[[VAL_7]], %[[VAL_8]] : index, index into memref<?x?xf64>
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_47]]] %[[VAL_38]], %[[VAL_6]], %[[VAL_8]] : index, index into memref<?x?xf64>
// CHECK:           %[[VAL_50:.*]], %[[VAL_51:.*]] = gpu.spmm_buffer_size async {{\[}}%[[VAL_49]]] %[[VAL_44]], %[[VAL_46]], %[[VAL_48]] : index
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = gpu.alloc async {{\[}}%[[VAL_51]]] (%[[VAL_50]]) : memref<?xi8>
// CHECK:           %[[VAL_54:.*]] = gpu.spmm async {{\[}}%[[VAL_53]]] %[[VAL_44]], %[[VAL_46]], %[[VAL_48]], %[[VAL_52]] : memref<?xi8>
// CHECK:           %[[VAL_55:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_54]]] %[[VAL_44]]
// CHECK:           %[[VAL_56:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_55]]] %[[VAL_46]]
// CHECK:           %[[VAL_57:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_56]]] %[[VAL_48]]
// CHECK:           %[[VAL_59:.*]] = gpu.dealloc async {{\[}}%[[VAL_57]]] %[[VAL_14]] : memref<?xindex>
// CHECK:           %[[VAL_60:.*]] = gpu.dealloc async {{\[}}%[[VAL_59]]] %[[VAL_19]] : memref<?xindex>
// CHECK:           %[[VAL_61:.*]] = gpu.dealloc async {{\[}}%[[VAL_60]]] %[[VAL_24]] : memref<?xf64>
// CHECK:           %[[VAL_62:.*]] = gpu.dealloc async {{\[}}%[[VAL_61]]] %[[VAL_52]] : memref<?xi8>
// CHECK:           %[[VAL_63:.*]] = gpu.dealloc async {{\[}}%[[VAL_62]]] %[[VAL_31]] : memref<?x?xf64>
// CHECK:           %[[VAL_64:.*]] = gpu.memcpy async {{\[}}%[[VAL_63]]] %[[VAL_34]], %[[VAL_38]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           %[[VAL_65:.*]] = gpu.dealloc async {{\[}}%[[VAL_64]]] %[[VAL_38]] : memref<?x?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_65]]]
// CHECK:           %[[VAL_66:.*]] = bufferization.to_tensor %[[VAL_34]] : memref<?x?xf64>
// CHECK:           return %[[VAL_66]] : tensor<?x?xf64>
// CHECK:         }
func.func @matmul(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64>, %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
      ins(%A, %B: tensor<?x?xf64, #CSR>, tensor<?x?xf64>)
      outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}
