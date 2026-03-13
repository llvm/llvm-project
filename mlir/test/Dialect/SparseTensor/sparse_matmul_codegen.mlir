// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:  --sparse-reinterpret-map --sparsification --sparse-tensor-codegen \
// RUN:  --canonicalize --cse | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

//
// Computes C = A x B with all matrices sparse (SpMSpM) in CSR.
//
// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:      %[[VAL_0:.*0]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_1:.*1]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_2:.*2]]: memref<?xf64>,
// CHECK-SAME:      %[[VAL_3:.*3]]: !sparse_tensor.storage_specifier
// CHECK-SAME:      %[[VAL_4:.*4]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_5:.*5]]: memref<?xindex>,
// CHECK-SAME:      %[[VAL_6:.*6]]: memref<?xf64>,
// CHECK-SAME:      %[[VAL_7:.*7]]: !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_13:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_14:.*]] = memref.alloc() : memref<16xindex>
// CHECK:           %[[VAL_15:.*]] = memref.cast %[[VAL_14]] : memref<16xindex> to memref<?xindex>
// CHECK:           %[[VAL_16:.*]] = memref.alloc() : memref<16xindex>
// CHECK:           %[[VAL_17:.*]] = memref.cast %[[VAL_16]] : memref<16xindex> to memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<16xf64>
// CHECK:           %[[VAL_19:.*]] = memref.cast %[[VAL_18]] : memref<16xf64> to memref<?xf64>
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_21:.*]] = sparse_tensor.storage_specifier.set %[[VAL_20]]  lvl_sz at 0 with %[[VAL_13]] : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_22:.*]] = sparse_tensor.storage_specifier.set %[[VAL_21]]  lvl_sz at 1 with %[[VAL_13]] : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.storage_specifier.get %[[VAL_22]]  pos_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_24:.*]], %[[VAL_25:.*]] = sparse_tensor.push_back %[[VAL_23]], %[[VAL_15]], %[[VAL_12]] : index, memref<?xindex>, index
// CHECK:           %[[VAL_26:.*]] = sparse_tensor.storage_specifier.set %[[VAL_22]]  pos_mem_sz at 1 with %[[VAL_25]] : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_27:.*]], %[[VAL_28:.*]] = sparse_tensor.push_back %[[VAL_25]], %[[VAL_24]], %[[VAL_12]], %[[VAL_13]] : index, memref<?xindex>, index, index
// CHECK:           %[[VAL_29:.*]] = sparse_tensor.storage_specifier.set %[[VAL_26]]  pos_mem_sz at 1 with %[[VAL_28]] : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_30:.*]] = memref.alloc() : memref<4xf64>
// CHECK:           %[[VAL_31:.*]] = memref.alloc() : memref<4xi1>
// CHECK:           %[[VAL_32:.*]] = memref.alloc() : memref<4xindex>
// CHECK:           %[[VAL_33:.*]] = memref.cast %[[VAL_32]] : memref<4xindex> to memref<?xindex>
// CHECK:           linalg.fill ins(%[[VAL_8]] : f64) outs(%[[VAL_30]] : memref<4xf64>)
// CHECK:           linalg.fill ins(%[[VAL_10]] : i1) outs(%[[VAL_31]] : memref<4xi1>)
// CHECK-DAG:       %[[VAL_34:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  pos_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_35:.*]] = memref.subview %[[VAL_0]][0] {{\[}}%[[VAL_34]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK-DAG:       %[[VAL_36:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  crd_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_37:.*]] = memref.subview %[[VAL_1]][0] {{\[}}%[[VAL_36]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK-DAG:       %[[VAL_38:.*]] = sparse_tensor.storage_specifier.get %[[VAL_3]]  val_mem_sz : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_39:.*]] = memref.subview %[[VAL_2]][0] {{\[}}%[[VAL_38]]] [1] : memref<?xf64> to memref<?xf64>
// CHECK-DAG:       %[[VAL_40:.*]] = sparse_tensor.storage_specifier.get %[[VAL_7]]  pos_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_41:.*]] = memref.subview %[[VAL_4]][0] {{\[}}%[[VAL_40]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK-DAG:       %[[VAL_42:.*]] = sparse_tensor.storage_specifier.get %[[VAL_7]]  crd_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_43:.*]] = memref.subview %[[VAL_5]][0] {{\[}}%[[VAL_42]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK-DAG:       %[[VAL_44:.*]] = sparse_tensor.storage_specifier.get %[[VAL_7]]  val_mem_sz : !sparse_tensor.storage_specifier
// CHECK-DAG:       %[[VAL_45:.*]] = memref.subview %[[VAL_6]][0] {{\[}}%[[VAL_44]]] [1] : memref<?xf64> to memref<?xf64>
// CHECK:           %[[VAL_46:.*]]:4 = scf.for %[[VAL_47:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_11]] iter_args(%[[VAL_48:.*]] = %[[VAL_27]], %[[VAL_49:.*]] = %[[VAL_17]], %[[VAL_50:.*]] = %[[VAL_19]], %[[VAL_51:.*]] = %[[VAL_29]]) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:             %[[VAL_52:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_47]]] : memref<?xindex>
// CHECK:             %[[VAL_53:.*]] = arith.addi %[[VAL_47]], %[[VAL_11]] : index
// CHECK:             %[[VAL_54:.*]] = memref.load %[[VAL_35]]{{\[}}%[[VAL_53]]] : memref<?xindex>
// CHECK:             %[[VAL_55:.*]] = scf.for %[[VAL_56:.*]] = %[[VAL_52]] to %[[VAL_54]] step %[[VAL_11]] iter_args(%[[VAL_57:.*]] = %[[VAL_12]]) -> (index) {
// CHECK:               %[[VAL_58:.*]] = memref.load %[[VAL_37]]{{\[}}%[[VAL_56]]] : memref<?xindex>
// CHECK:               %[[VAL_59:.*]] = memref.load %[[VAL_39]]{{\[}}%[[VAL_56]]] : memref<?xf64>
// CHECK:               %[[VAL_60:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_58]]] : memref<?xindex>
// CHECK:               %[[VAL_61:.*]] = arith.addi %[[VAL_58]], %[[VAL_11]] : index
// CHECK:               %[[VAL_62:.*]] = memref.load %[[VAL_41]]{{\[}}%[[VAL_61]]] : memref<?xindex>
// CHECK:               %[[VAL_63:.*]] = scf.for %[[VAL_64:.*]] = %[[VAL_60]] to %[[VAL_62]] step %[[VAL_11]] iter_args(%[[VAL_65:.*]] = %[[VAL_57]]) -> (index) {
// CHECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_43]]{{\[}}%[[VAL_64]]] : memref<?xindex>
// CHECK:                 %[[VAL_67:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_66]]] : memref<4xf64>
// CHECK:                 %[[VAL_68:.*]] = memref.load %[[VAL_45]]{{\[}}%[[VAL_64]]] : memref<?xf64>
// CHECK:                 %[[VAL_69:.*]] = arith.mulf %[[VAL_59]], %[[VAL_68]] : f64
// CHECK:                 %[[VAL_70:.*]] = arith.addf %[[VAL_67]], %[[VAL_69]] : f64
// CHECK:                 %[[VAL_71:.*]] = memref.load %[[VAL_31]]{{\[}}%[[VAL_66]]] : memref<4xi1>
// CHECK:                 %[[VAL_72:.*]] = arith.cmpi eq, %[[VAL_71]], %[[VAL_10]] : i1
// CHECK:                 %[[VAL_73:.*]] = scf.if %[[VAL_72]] -> (index) {
// CHECK:                   memref.store %[[VAL_9]], %[[VAL_31]]{{\[}}%[[VAL_66]]] : memref<4xi1>
// CHECK:                   memref.store %[[VAL_66]], %[[VAL_32]]{{\[}}%[[VAL_65]]] : memref<4xindex>
// CHECK:                   %[[VAL_74:.*]] = arith.addi %[[VAL_65]], %[[VAL_11]] : index
// CHECK:                   scf.yield %[[VAL_74]] : index
// CHECK:                 } else {
// CHECK:                   scf.yield %[[VAL_65]] : index
// CHECK:                 }
// CHECK:                 memref.store %[[VAL_70]], %[[VAL_30]]{{\[}}%[[VAL_66]]] : memref<4xf64>
// CHECK:                 scf.yield %[[VAL_73]] : index
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               scf.yield %[[VAL_63]] : index
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             sparse_tensor.sort  hybrid_quick_sort %[[VAL_55]], %[[VAL_33]]
// CHECK:             %[[VAL_75:.*]]:4 = scf.for %[[VAL_76:.*]] = %[[VAL_12]] to %[[VAL_55]] step %[[VAL_11]] iter_args(%[[VAL_77:.*]] = %[[VAL_48]], %[[VAL_78:.*]] = %[[VAL_49]], %[[VAL_79:.*]] = %[[VAL_50]], %[[VAL_80:.*]] = %[[VAL_51]]) -> (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:               %[[VAL_81:.*]] = memref.load %[[VAL_32]]{{\[}}%[[VAL_76]]] : memref<4xindex>
// CHECK:               %[[VAL_82:.*]] = memref.load %[[VAL_30]]{{\[}}%[[VAL_81]]] : memref<4xf64>
// CHECK:               %[[VAL_83:.*]]:4 = func.call @_insert_dense_compressed_4_4_f64_0_0(%[[VAL_77]], %[[VAL_78]], %[[VAL_79]], %[[VAL_80]], %[[VAL_47]], %[[VAL_81]], %[[VAL_82]]) : (memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:               memref.store %[[VAL_8]], %[[VAL_30]]{{\[}}%[[VAL_81]]] : memref<4xf64>
// CHECK:               memref.store %[[VAL_10]], %[[VAL_31]]{{\[}}%[[VAL_81]]] : memref<4xi1>
// CHECK:               scf.yield %[[VAL_83]]#0, %[[VAL_83]]#1, %[[VAL_83]]#2, %[[VAL_83]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:             }
// CHECK:             scf.yield %[[VAL_84:.*]]#0, %[[VAL_84]]#1, %[[VAL_84]]#2, %[[VAL_84]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.dealloc %[[VAL_30]] : memref<4xf64>
// CHECK:           memref.dealloc %[[VAL_31]] : memref<4xi1>
// CHECK:           memref.dealloc %[[VAL_32]] : memref<4xindex>
// CHECK:           %[[VAL_85:.*]] = sparse_tensor.storage_specifier.get %[[VAL_86:.*]]#3  pos_mem_sz at 1 : !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_87:.*]] = memref.load %[[VAL_86]]#0{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:           %[[VAL_88:.*]] = scf.for %[[VAL_89:.*]] = %[[VAL_11]] to %[[VAL_85]] step %[[VAL_11]] iter_args(%[[VAL_90:.*]] = %[[VAL_87]]) -> (index) {
// CHECK:             %[[VAL_91:.*]] = memref.load %[[VAL_86]]#0{{\[}}%[[VAL_89]]] : memref<?xindex>
// CHECK:             %[[VAL_92:.*]] = arith.cmpi eq, %[[VAL_91]], %[[VAL_12]] : index
// CHECK:             %[[VAL_93:.*]] = arith.select %[[VAL_92]], %[[VAL_90]], %[[VAL_91]] : index
// CHECK:             scf.if %[[VAL_92]] {
// CHECK:               memref.store %[[VAL_90]], %[[VAL_86]]#0{{\[}}%[[VAL_89]]] : memref<?xindex>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_93]] : index
// CHECK:           }
// CHECK:           return %[[VAL_86]]#0, %[[VAL_86]]#1, %[[VAL_86]]#2, %[[VAL_86]]#3 : memref<?xindex>, memref<?xindex>, memref<?xf64>, !sparse_tensor.storage_specifier
// CHECK:         }
func.func @matmul(%A: tensor<4x8xf64, #CSR>,
                  %B: tensor<8x4xf64, #CSR>) -> tensor<4x4xf64, #CSR> {
  %C = tensor.empty() : tensor<4x4xf64, #CSR>
  %D = linalg.matmul
    ins(%A, %B: tensor<4x8xf64, #CSR>, tensor<8x4xf64, #CSR>)
       outs(%C: tensor<4x4xf64, #CSR>) -> tensor<4x4xf64, #CSR>
  return %D: tensor<4x4xf64, #CSR>
}
