// RUN: mlir-opt %s --linalg-generalize-named-ops --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#NV_24 = #sparse_tensor.encoding<{
  map = ( i, j ) ->
  ( i            : dense,
    j floordiv 4 : dense,
    j mod 4      : structured[2, 4]
  )
}>

// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:      %[[VAL_0:.*0]]: tensor<?x?xf16>,
// CHECK-SAME:      %[[VAL_1:.*1]]: tensor<?x?xf16>,
// CHECK-SAME:      %[[VAL_2:.*2]]: tensor<?x?xf16>) -> tensor<?x?xf16> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = bufferization.to_memref %[[VAL_0]] : memref<?x?xf16>
// CHECK:           %[[VAL_6:.*]] = gpu.wait async
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_5]], %[[VAL_3]] : memref<?x?xf16>
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_5]], %[[VAL_4]] : memref<?x?xf16>
// CHECK:           %[[VAL_9:.*]], %[[VAL_10:.*]] = gpu.alloc async {{\[}}%[[VAL_6]]] (%[[VAL_7]], %[[VAL_8]]) : memref<?x?xf16>
// CHECK:           %[[VAL_11:.*]] = gpu.memcpy async {{\[}}%[[VAL_10]]] %[[VAL_9]], %[[VAL_5]] : memref<?x?xf16>, memref<?x?xf16>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_1]] : memref<?x?xf16>
// CHECK:           %[[VAL_13:.*]] = gpu.wait async
// CHECK:           %[[VAL_14:.*]] = memref.dim %[[VAL_12]], %[[VAL_3]] : memref<?x?xf16>
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_12]], %[[VAL_4]] : memref<?x?xf16>
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = gpu.alloc async {{\[}}%[[VAL_13]]] (%[[VAL_14]], %[[VAL_15]]) : memref<?x?xf16>
// CHECK:           %[[VAL_18:.*]] = gpu.memcpy async {{\[}}%[[VAL_17]]] %[[VAL_16]], %[[VAL_12]] : memref<?x?xf16>, memref<?x?xf16>
// CHECK:           %[[VAL_19:.*]] = bufferization.to_memref %[[VAL_2]] : memref<?x?xf16>
// CHECK:           %[[VAL_20:.*]] = gpu.wait async
// CHECK:           %[[VAL_21:.*]] = memref.dim %[[VAL_19]], %[[VAL_3]] : memref<?x?xf16>
// CHECK:           %[[VAL_22:.*]] = memref.dim %[[VAL_19]], %[[VAL_4]] : memref<?x?xf16>
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = gpu.alloc async {{\[}}%[[VAL_20]]] (%[[VAL_21]], %[[VAL_22]]) : memref<?x?xf16>
// CHECK:           %[[VAL_25:.*]] = gpu.memcpy async {{\[}}%[[VAL_24]]] %[[VAL_23]], %[[VAL_19]] : memref<?x?xf16>, memref<?x?xf16>
// CHECK:           gpu.wait {{\[}}%[[VAL_11]], %[[VAL_18]], %[[VAL_25]]]
// CHECK:           %[[VAL_26:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?x?xf16>
// CHECK:           %[[VAL_27:.*]] = memref.dim %[[VAL_16]], %[[VAL_3]] : memref<?x?xf16>
// CHECK:           %[[VAL_28:.*]] = memref.dim %[[VAL_23]], %[[VAL_4]] : memref<?x?xf16>
// CHECK:           %[[VAL_29:.*]] = gpu.wait async
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = gpu.create_2to4_spmat async {{\[}}%[[VAL_29]]]{{{.*}}} %[[VAL_26]], %[[VAL_27]], %[[VAL_9]] : memref<?x?xf16>
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_31]]] %[[VAL_16]], %[[VAL_27]], %[[VAL_28]] : index, index into memref<?x?xf16>
// CHECK:           %[[VAL_34:.*]], %[[VAL_35:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_33]]] %[[VAL_23]], %[[VAL_26]], %[[VAL_28]] : index, index into memref<?x?xf16>
// CHECK:           %[[VAL_36:.*]]:3, %[[VAL_37:.*]] = gpu.spmm_buffer_size async {{\[}}%[[VAL_35]]] %[[VAL_30]], %[[VAL_32]], %[[VAL_34]] : index, index, index into f16
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = gpu.alloc async {{\[}}%[[VAL_37]]] (%[[VAL_36]]#0) : memref<?xi8>
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = gpu.alloc async {{\[}}%[[VAL_39]]] (%[[VAL_36]]#1) : memref<?xi8>
// CHECK:           %[[VAL_42:.*]], %[[VAL_43:.*]] = gpu.alloc async {{\[}}%[[VAL_41]]] (%[[VAL_36]]#2) : memref<?xi8>
// CHECK:           %[[VAL_44:.*]] = gpu.spmm async {{\[}}%[[VAL_43]]] %[[VAL_30]], %[[VAL_32]], %[[VAL_34]], %[[VAL_38]], %[[VAL_40]], %[[VAL_42]] : memref<?xi8>, memref<?xi8>, memref<?xi8> into f16
// CHECK:           %[[VAL_45:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_44]]] %[[VAL_30]]
// CHECK:           %[[VAL_46:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_45]]] %[[VAL_32]]
// CHECK:           %[[VAL_47:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_46]]] %[[VAL_34]]
// CHECK:           %[[VAL_48:.*]] = gpu.dealloc async {{\[}}%[[VAL_47]]] %[[VAL_38]] : memref<?xi8>
// CHECK:           %[[VAL_49:.*]] = gpu.dealloc async {{\[}}%[[VAL_48]]] %[[VAL_40]] : memref<?xi8>
// CHECK:           %[[VAL_50:.*]] = gpu.dealloc async {{\[}}%[[VAL_49]]] %[[VAL_42]] : memref<?xi8>
// CHECK:           %[[VAL_51:.*]] = gpu.dealloc async {{\[}}%[[VAL_50]]] %[[VAL_9]] : memref<?x?xf16>
// CHECK:           %[[VAL_52:.*]] = gpu.dealloc async {{\[}}%[[VAL_51]]] %[[VAL_16]] : memref<?x?xf16>
// CHECK:           %[[VAL_53:.*]] = gpu.memcpy async {{\[}}%[[VAL_52]]] %[[VAL_19]], %[[VAL_23]] : memref<?x?xf16>, memref<?x?xf16>
// CHECK:           %[[VAL_54:.*]] = gpu.dealloc async {{\[}}%[[VAL_53]]] %[[VAL_23]] : memref<?x?xf16>
// CHECK:           gpu.wait {{\[}}%[[VAL_54]]]
// CHECK:           %[[VAL_55:.*]] = bufferization.to_tensor %[[VAL_19]] : memref<?x?xf16>
// CHECK:           return %[[VAL_55]] : tensor<?x?xf16>
// CHECK:         }
module {
  func.func @matmul(%Ad: tensor<?x?xf16>,
                    %B: tensor<?x?xf16>,
		    %Cin: tensor<?x?xf16>) -> tensor<?x?xf16> {
    %A = sparse_tensor.convert %Ad : tensor<?x?xf16> to tensor<?x?xf16, #NV_24>
    %C = linalg.matmul
      ins(%A, %B: tensor<?x?xf16, #NV_24>, tensor<?x?xf16>)
      outs(%Cin: tensor<?x?xf16>) -> tensor<?x?xf16>
    return %C : tensor<?x?xf16>
  }
}
