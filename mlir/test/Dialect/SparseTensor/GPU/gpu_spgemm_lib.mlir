// RUN: mlir-opt %s --linalg-generalize-named-ops --sparse-gpu-codegen="num-threads=0" | FileCheck %s

#CSR = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>

// CHECK-LABEL: func.func @matmulCSR(
// CHECK-SAME:      %[[VAL_0:.*0]]: tensor<8x8xf32, #{{.*}}>,
// CHECK-SAME:      %[[VAL_1:.*1]]: tensor<8x8xf32, #{{.*}}>) -> tensor<8x8xf32, #{{.*}}> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.number_of_entries %[[VAL_1]] : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 1 : index} : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<8x8xf32, #{{.*}}>
// CHECK:           %[[VAL_14:.*]] = gpu.wait async
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_8]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = gpu.alloc async {{\[}}%[[VAL_14]]] (%[[VAL_15]]) : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = gpu.memcpy async {{\[}}%[[VAL_17]]] %[[VAL_16]], %[[VAL_8]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = gpu.wait async
// CHECK:           %[[VAL_20:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = gpu.alloc async {{\[}}%[[VAL_19]]] (%[[VAL_20]]) : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = gpu.memcpy async {{\[}}%[[VAL_22]]] %[[VAL_21]], %[[VAL_9]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = gpu.wait async
// CHECK:           %[[VAL_25:.*]] = memref.dim %[[VAL_10]], %[[VAL_3]] : memref<?xf32>
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = gpu.alloc async {{\[}}%[[VAL_24]]] (%[[VAL_25]]) : memref<?xf32>
// CHECK:           %[[VAL_28:.*]] = gpu.memcpy async {{\[}}%[[VAL_27]]] %[[VAL_26]], %[[VAL_10]] : memref<?xf32>, memref<?xf32>
// CHECK:           %[[VAL_29:.*]] = gpu.wait async
// CHECK:           %[[VAL_30:.*]] = memref.dim %[[VAL_11]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = gpu.alloc async {{\[}}%[[VAL_29]]] (%[[VAL_30]]) : memref<?xindex>
// CHECK:           %[[VAL_33:.*]] = gpu.memcpy async {{\[}}%[[VAL_32]]] %[[VAL_31]], %[[VAL_11]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_34:.*]] = gpu.wait async
// CHECK:           %[[VAL_35:.*]] = memref.dim %[[VAL_12]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = gpu.alloc async {{\[}}%[[VAL_34]]] (%[[VAL_35]]) : memref<?xindex>
// CHECK:           %[[VAL_38:.*]] = gpu.memcpy async {{\[}}%[[VAL_37]]] %[[VAL_36]], %[[VAL_12]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_39:.*]] = gpu.wait async
// CHECK:           %[[VAL_40:.*]] = memref.dim %[[VAL_13]], %[[VAL_3]] : memref<?xf32>
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = gpu.alloc async {{\[}}%[[VAL_39]]] (%[[VAL_40]]) : memref<?xf32>
// CHECK:           %[[VAL_43:.*]] = gpu.memcpy async {{\[}}%[[VAL_42]]] %[[VAL_41]], %[[VAL_13]] : memref<?xf32>, memref<?xf32>
// CHECK:           gpu.wait {{\[}}%[[VAL_18]], %[[VAL_23]], %[[VAL_28]], %[[VAL_33]], %[[VAL_38]], %[[VAL_43]]]
// CHECK:           %[[VAL_44:.*]] = gpu.wait async
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = gpu.create_csr async {{\[}}%[[VAL_44]]] %[[VAL_2]], %[[VAL_2]], %[[VAL_6]], %[[VAL_16]], %[[VAL_21]], %[[VAL_26]] : memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = gpu.create_csr async {{\[}}%[[VAL_46]]] %[[VAL_2]], %[[VAL_2]], %[[VAL_7]], %[[VAL_31]], %[[VAL_36]], %[[VAL_41]] : memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:           %[[VAL_49:.*]], %[[VAL_50:.*]] = gpu.alloc async {{\[}}%[[VAL_48]]] (%[[VAL_4]]) : memref<?xindex>
// CHECK:           %[[VAL_51:.*]], %[[VAL_52:.*]] = gpu.alloc async {{\[}}%[[VAL_50]]] (%[[VAL_3]]) : memref<?xindex>
// CHECK:           %[[VAL_53:.*]], %[[VAL_54:.*]] = gpu.alloc async {{\[}}%[[VAL_52]]] (%[[VAL_3]]) : memref<?xf32>
// CHECK:           %[[VAL_55:.*]], %[[VAL_56:.*]] = gpu.create_csr async {{\[}}%[[VAL_54]]] %[[VAL_2]], %[[VAL_2]], %[[VAL_3]], %[[VAL_49]], %[[VAL_51]], %[[VAL_53]] : memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:           %[[VAL_57:.*]], %[[VAL_58:.*]] = gpu.spgemm_create_descr async {{\[}}%[[VAL_56]]]
// CHECK:           %[[VAL_59:.*]], %[[VAL_60:.*]] = gpu.spgemm_work_estimation_or_compute async {{\[}}%[[VAL_58]]]{ WORK_ESTIMATION} %[[VAL_45]], %[[VAL_47]], %[[VAL_55]], %[[VAL_57]], %[[VAL_3]], %[[VAL_53]] : f32 into memref<?xf32>
// CHECK:           %[[VAL_61:.*]], %[[VAL_62:.*]] = gpu.alloc async {{\[}}%[[VAL_60]]] (%[[VAL_59]]) : memref<?xi8>
// CHECK:           %[[VAL_63:.*]], %[[VAL_64:.*]] = gpu.spgemm_work_estimation_or_compute async {{\[}}%[[VAL_62]]]{ WORK_ESTIMATION} %[[VAL_45]], %[[VAL_47]], %[[VAL_55]], %[[VAL_57]], %[[VAL_59]], %[[VAL_61]] : f32 into memref<?xi8>
// CHECK:           %[[VAL_65:.*]], %[[VAL_66:.*]] = gpu.spgemm_work_estimation_or_compute async {{\[}}%[[VAL_64]]]{ COMPUTE} %[[VAL_45]], %[[VAL_47]], %[[VAL_55]], %[[VAL_57]], %[[VAL_3]], %[[VAL_53]] : f32 into memref<?xf32>
// CHECK:           %[[VAL_67:.*]], %[[VAL_68:.*]] = gpu.alloc async {{\[}}%[[VAL_66]]] (%[[VAL_65]]) : memref<?xi8>
// CHECK:           %[[VAL_69:.*]], %[[VAL_70:.*]] = gpu.spgemm_work_estimation_or_compute async {{\[}}%[[VAL_68]]]{ COMPUTE} %[[VAL_45]], %[[VAL_47]], %[[VAL_55]], %[[VAL_57]], %[[VAL_65]], %[[VAL_67]] : f32 into memref<?xi8>
// CHECK:           %[[VAL_71:.*]], %[[VAL_72:.*]], %[[VAL_73:.*]], %[[VAL_74:.*]] = gpu.spmat_get_size async {{\[}}%[[VAL_70]]] %[[VAL_55]]
// CHECK:           %[[VAL_75:.*]], %[[VAL_76:.*]] = gpu.alloc async {{\[}}%[[VAL_74]]] (%[[VAL_73]]) : memref<?xindex>
// CHECK:           %[[VAL_77:.*]], %[[VAL_78:.*]] = gpu.alloc async {{\[}}%[[VAL_76]]] (%[[VAL_73]]) : memref<?xf32>
// CHECK:           %[[VAL_79:.*]] = gpu.set_csr_pointers async {{\[}}%[[VAL_78]]] %[[VAL_55]], %[[VAL_49]], %[[VAL_75]], %[[VAL_77]] : memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:           %[[VAL_80:.*]] = gpu.spgemm_copy async {{\[}}%[[VAL_79]]] %[[VAL_45]], %[[VAL_47]], %[[VAL_55]], %[[VAL_57]] : f32
// CHECK:           %[[VAL_81:.*]] = memref.alloc(%[[VAL_4]]) : memref<?xindex>
// CHECK:           %[[VAL_82:.*]] = memref.alloc(%[[VAL_73]]) : memref<?xindex>
// CHECK:           %[[VAL_83:.*]] = memref.alloc(%[[VAL_73]]) : memref<?xf32>
// CHECK:           %[[VAL_84:.*]] = gpu.spgemm_destroy_descr async {{\[}}%[[VAL_80]]] %[[VAL_57]]
// CHECK:           %[[VAL_85:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_84]]] %[[VAL_45]]
// CHECK:           %[[VAL_86:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_85]]] %[[VAL_47]]
// CHECK:           %[[VAL_87:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_86]]] %[[VAL_55]]
// CHECK:           %[[VAL_88:.*]] = gpu.memcpy async {{\[}}%[[VAL_87]]] %[[VAL_81]], %[[VAL_49]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_89:.*]] = gpu.memcpy async {{\[}}%[[VAL_88]]] %[[VAL_82]], %[[VAL_75]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_90:.*]] = gpu.memcpy async {{\[}}%[[VAL_89]]] %[[VAL_83]], %[[VAL_77]] : memref<?xf32>, memref<?xf32>
// CHECK:           %[[VAL_91:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_92:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_93:.*]] = gpu.dealloc async {{.*}} : memref<?xf32>
// CHECK:           %[[VAL_94:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_95:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_96:.*]] = gpu.dealloc async {{.*}} : memref<?xf32>
// CHECK:           %[[VAL_97:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_98:.*]] = gpu.dealloc async {{.*}} : memref<?xindex>
// CHECK:           %[[VAL_99:.*]] = gpu.dealloc async {{.*}} : memref<?xf32>
// CHECK:           %[[VAL_a0:.*]] = gpu.dealloc async {{.*}} : memref<?xi8>
// CHECK:           %[[VAL_a1:.*]] = gpu.dealloc async {{.*}} : memref<?xi8>
// CHECK:           gpu.wait [%[[VAL_a1]]]
// CHECK:           %[[VAL_a2:.*]] = bufferization.to_tensor %[[VAL_83]] : memref<?xf32>
// CHECK:           %[[VAL_a3:.*]] = bufferization.to_tensor %[[VAL_81]] : memref<?xindex>
// CHECK:           %[[VAL_a4:.*]] = bufferization.to_tensor %[[VAL_82]] : memref<?xindex>
// CHECK:           %[[VAL_a5:.*]] = sparse_tensor.assemble (%[[VAL_a3]], %[[VAL_a4]]), %[[VAL_a2]] : (tensor<?xindex>, tensor<?xindex>), tensor<?xf32> to tensor<8x8xf32, #{{.*}}>
// CHECK:           return %[[VAL_a5]] : tensor<8x8xf32, #{{.*}}>
// CHECK:         }
func.func @matmulCSR(%A: tensor<8x8xf32, #CSR>,
                     %B: tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR> {
  %init = tensor.empty() : tensor<8x8xf32, #CSR>
  %C = linalg.matmul
    ins(%A, %B: tensor<8x8xf32, #CSR>,
                tensor<8x8xf32, #CSR>)
    outs(%init: tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR>
  return %C: tensor<8x8xf32, #CSR>
}
