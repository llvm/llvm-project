// RUN: mlir-opt %s -split-input-file --sparse-buffer-rewrite  --canonicalize --cse | FileCheck %s

// CHECK-LABEL: func @sparse_push_back(
//  CHECK-SAME: %[[A:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//       CHECK: %[[P1:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[S2:.*]] = arith.addi %[[A]], %[[C1]] : index
//       CHECK: %[[T:.*]] = arith.cmpi ugt, %[[S2]], %[[P1]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:  %[[P2:.*]] = arith.muli %[[P1]], %[[C2]]
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P2]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: memref.store %[[C]], %[[M]]{{\[}}%[[A]]]
//       CHECK: return %[[M]], %[[S2]]
func.func @sparse_push_back(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_n(
//  CHECK-SAME: %[[S1:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64,
//  CHECK-SAME: %[[D:.*]]: index) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
//       CHECK: %[[P1:.*]] = memref.dim %[[B]], %[[C0]]
//       CHECK: %[[S2:.*]] = arith.addi %[[S1]], %[[D]] : index
//       CHECK: %[[T:.*]] = arith.cmpi ugt, %[[S2]], %[[P1]]
//       CHECK: %[[M:.*]] = scf.if %[[T]] -> (memref<?xf64>) {
//       CHECK:   %[[P2:.*]] = scf.while (%[[I:.*]] = %[[P1]]) : (index) -> index {
//       CHECK:     %[[P3:.*]] = arith.muli %[[I]], %[[C2]] : index
//       CHECK:     %[[T2:.*]] = arith.cmpi ugt, %[[S2]], %[[P3]] : index
//       CHECK:     scf.condition(%[[T2]]) %[[P3]] : index
//       CHECK:   } do {
//       CHECK:     ^bb0(%[[I2:.*]]: index):
//       CHECK:     scf.yield %[[I2]] : index
//       CHECK:   }
//       CHECK:  %[[M2:.*]] = memref.realloc %[[B]](%[[P2]])
//       CHECK:  scf.yield %[[M2]] : memref<?xf64>
//       CHECK: } else {
//       CHECK:  scf.yield %[[B]] : memref<?xf64>
//       CHECK: }
//       CHECK: %[[S:.*]] = memref.subview %[[M]]{{\[}}%[[S1]]] {{\[}}%[[D]]] [1]
//       CHECK: linalg.fill ins(%[[C]] : f64) outs(%[[S]]
//       CHECK: return %[[M]], %[[S2]] : memref<?xf64>, index
func.func @sparse_push_back_n(%arg0: index, %arg1: memref<?xf64>, %arg2: f64, %arg3: index) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back %arg0, %arg1, %arg2, %arg3 : index, memref<?xf64>, f64, index
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL: func @sparse_push_back_inbound(
//  CHECK-SAME: %[[S1:.*]]: index,
//  CHECK-SAME: %[[B:.*]]: memref<?xf64>,
//  CHECK-SAME: %[[C:.*]]: f64) -> (memref<?xf64>, index) {
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[S2:.*]] = arith.addi %[[S1]], %[[C1]]
//       CHECK: memref.store %[[C]], %[[B]]{{\[}}%[[S1]]]
//       CHECK: return %[[B]], %[[S2]] : memref<?xf64>, index
func.func @sparse_push_back_inbound(%arg0: index, %arg1: memref<?xf64>, %arg2: f64) -> (memref<?xf64>, index) {
  %0:2 = sparse_tensor.push_back inbounds %arg0, %arg1, %arg2 : index, memref<?xf64>, f64
  return %0#0, %0#1 : memref<?xf64>, index
}

// -----

// CHECK-LABEL:   func.func private @_sparse_partition_1_i8_f32_index(
// CHECK-SAME:    %[[VAL_0:.*0]]: index,
// CHECK-SAME:    %[[VAL_1:.*1]]: index,
// CHECK-SAME:    %[[VAL_2:.*2]]: memref<?xi8>,
// CHECK-SAME:    %[[VAL_3:.*3]]: memref<?xf32>,
// CHECK-SAME:    %[[VAL_4:.*4]]: memref<?xindex>) -> index {
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant -1
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]]
// CHECK:           %[[VAL_8:.*]] = arith.shrui %[[VAL_7]], %[[VAL_5]]
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_1]], %[[VAL_5]]
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_8]]]
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ult, %[[VAL_10]], %[[VAL_11]]
// CHECK:           %[[VAL_13:.*]] = scf.if %[[VAL_12]] -> (index) {
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_15:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:             %[[VAL_16:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_15]]
// CHECK:             %[[VAL_17:.*]] = scf.if %[[VAL_16]] -> (index) {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_8]]]
// CHECK:               %[[VAL_20:.*]] = arith.cmpi ult, %[[VAL_18]], %[[VAL_19]]
// CHECK:               %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_8]], %[[VAL_9]]
// CHECK:               scf.yield %[[VAL_21]]
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_0]]
// CHECK:             }
// CHECK:             scf.yield %[[VAL_22:.*]]
// CHECK:           } else {
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_24:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_8]]]
// CHECK:             %[[VAL_25:.*]] = arith.cmpi ult, %[[VAL_23]], %[[VAL_24]]
// CHECK:             %[[VAL_26:.*]] = scf.if %[[VAL_25]] -> (index) {
// CHECK:               %[[VAL_27:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_28:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_27]], %[[VAL_28]]
// CHECK:               %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_0]], %[[VAL_9]]
// CHECK:               scf.yield %[[VAL_30]]
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_8]]
// CHECK:             }
// CHECK:             scf.yield %[[VAL_31:.*]]
// CHECK:           }
// CHECK:           %[[VAL_32:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_13:.*]]
// CHECK:           scf.if %[[VAL_32]] {
// CHECK:             %[[VAL_34:.*]] = memref.load %[[VAL_2]]{{\[}}
// CHECK:             %[[VAL_35:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_8]]]
// CHECK:             memref.store %[[VAL_35]], %[[VAL_2]]
// CHECK:             memref.store %[[VAL_34]], %[[VAL_2]]{{\[}}%[[VAL_8]]]
// CHECK:             %[[VAL_36:.*]] = memref.load %[[VAL_3]]
// CHECK:             %[[VAL_37:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_8]]]
// CHECK:             memref.store %[[VAL_37]], %[[VAL_3]]
// CHECK:             memref.store %[[VAL_36]], %[[VAL_3]]{{\[}}%[[VAL_8]]]
// CHECK:             %[[VAL_38:.*]] = memref.load %[[VAL_4]]
// CHECK:             %[[VAL_39:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_8]]]
// CHECK:             memref.store %[[VAL_39]], %[[VAL_4]]
// CHECK:             memref.store %[[VAL_38]], %[[VAL_4]]{{\[}}%[[VAL_8]]]
// CHECK:           }
// CHECK:           %[[VAL_40:.*]]:3 = scf.while (%[[VAL_41:.*]] = %[[VAL_0]], %[[VAL_42:.*]] = %[[VAL_9]], %[[VAL_43:.*]] = %[[VAL_8]])
// CHECK:             %[[VAL_44:.*]] = arith.cmpi ult, %[[VAL_41]], %[[VAL_42]]
// CHECK:             scf.condition(%[[VAL_44]]) %[[VAL_41]], %[[VAL_42]], %[[VAL_43]]
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_45:.*]]: index, %[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index):
// CHECK:             %[[VAL_48:.*]] = scf.while (%[[VAL_49:.*]] = %[[VAL_45]]) : (index) -> index {
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_49]]]
// CHECK:               %[[VAL_51:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_47]]]
// CHECK:               %[[VAL_52:.*]] = arith.cmpi ult, %[[VAL_50]], %[[VAL_51]]
// CHECK:               scf.condition(%[[VAL_52]]) %[[VAL_49]]
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_53:.*]]: index):
// CHECK:               %[[VAL_54:.*]] = arith.addi %[[VAL_53]], %[[VAL_5]]
// CHECK:               scf.yield %[[VAL_54]]
// CHECK:             }
// CHECK:             %[[VAL_55:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_56:.*]]]
// CHECK:             %[[VAL_57:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_47]]]
// CHECK:             %[[VAL_58:.*]] = arith.cmpi eq, %[[VAL_55]], %[[VAL_57]]
// CHECK:             %[[VAL_59:.*]] = scf.while (%[[VAL_60:.*]] = %[[VAL_46]]) : (index) -> index {
// CHECK:               %[[VAL_61:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_47]]]
// CHECK:               %[[VAL_62:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_60]]]
// CHECK:               %[[VAL_63:.*]] = arith.cmpi ult, %[[VAL_61]], %[[VAL_62]]
// CHECK:               scf.condition(%[[VAL_63]]) %[[VAL_60]]
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_64:.*]]: index):
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_64]], %[[VAL_6]]
// CHECK:               scf.yield %[[VAL_65]]
// CHECK:             }
// CHECK:             %[[VAL_66:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_67:.*]]]
// CHECK:             %[[VAL_68:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_47]]]
// CHECK:             %[[VAL_69:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_68]]
// CHECK:             %[[VAL_70:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_67]]
// CHECK:             %[[VAL_71:.*]]:3 = scf.if %[[VAL_70]] -> (index, index, index) {
// CHECK:               %[[VAL_72:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_56]]]
// CHECK:               %[[VAL_73:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_67]]]
// CHECK:               memref.store %[[VAL_73]], %[[VAL_2]]{{\[}}%[[VAL_56]]]
// CHECK:               memref.store %[[VAL_72]], %[[VAL_2]]{{\[}}%[[VAL_67]]]
// CHECK:               %[[VAL_74:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_56]]]
// CHECK:               %[[VAL_75:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_67]]]
// CHECK:               memref.store %[[VAL_75]], %[[VAL_3]]{{\[}}%[[VAL_56]]]
// CHECK:               memref.store %[[VAL_74]], %[[VAL_3]]{{\[}}%[[VAL_67]]]
// CHECK:               %[[VAL_76:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_56]]]
// CHECK:               %[[VAL_77:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_67]]]
// CHECK:               memref.store %[[VAL_77]], %[[VAL_4]]{{\[}}%[[VAL_56]]]
// CHECK:               memref.store %[[VAL_76]], %[[VAL_4]]{{\[}}%[[VAL_67]]]
// CHECK:               %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_56]], %[[VAL_47]]
// CHECK:               %[[VAL_79:.*]] = scf.if %[[VAL_78]] -> (index) {
// CHECK:                 scf.yield %[[VAL_67]]
// CHECK:               } else {
// CHECK:                 %[[VAL_80:.*]] = arith.cmpi eq, %[[VAL_67]], %[[VAL_47]]
// CHECK:                 %[[VAL_81:.*]] = arith.select %[[VAL_80]], %[[VAL_56]], %[[VAL_47]]
// CHECK:                 scf.yield %[[VAL_81]]
// CHECK:               }
// CHECK:               %[[VAL_82:.*]] = arith.andi %[[VAL_58]], %[[VAL_69]]
// CHECK:               %[[VAL_83:.*]]:2 = scf.if %[[VAL_82]] -> (index, index) {
// CHECK:                 %[[VAL_84:.*]] = arith.addi %[[VAL_56]], %[[VAL_5]]
// CHECK:                 %[[VAL_85:.*]] = arith.subi %[[VAL_67]], %[[VAL_5]]
// CHECK:                 scf.yield %[[VAL_84]], %[[VAL_85]]
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_56]], %[[VAL_67]]
// CHECK:               }
// CHECK:               scf.yield %[[VAL_86:.*]]#0, %[[VAL_86]]#1, %[[VAL_87:.*]]
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_56]], %[[VAL_67]], %[[VAL_47]]
// CHECK:             }
// CHECK:             scf.yield %[[VAL_88:.*]]#0, %[[VAL_88]]#1, %[[VAL_88]]#2
// CHECK:           }
// CHECK:           return %[[VAL_89:.*]]#2
// CHECK:         }

// CHECK-LABEL:   func.func private @_sparse_qsort_1_i8_f32_index(
// CHECK-SAME:                                                   %[[L:arg0]]: index,
// CHECK-SAME:                                                   %[[H:.*]]: index,
// CHECK-SAME:                                                   %[[X0:.*]]: memref<?xi8>,
// CHECK-SAME:                                                   %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                                   %[[Y1:.*]]: memref<?xindex>) {
// CHECK:           %[[C1:.*]] = arith.constant 1
// CHECK:           scf.while (%[[L2:.*]] = %[[L]], %[[H2:.*]] = %[[H]])
// CHECK:             %[[Lb:.*]] = arith.addi %[[L2]], %[[C1]]
// CHECK:             %[[COND:.*]] = arith.cmpi ult, %[[Lb]], %[[H2]]
// CHECK:             scf.condition(%[[COND]]) %[[L2]], %[[H2]]
// CHECK:           } do {
// CHECK:           ^bb0(%[[L3:.*]]: index, %[[H3:.*]]: index)
// CHECK:             %[[P:.*]] = func.call @_sparse_partition_1_i8_f32_index(%[[L3]], %[[H3]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:             %[[PP1:.*]] = arith.addi %[[P]], %[[C1]] : index
// CHECK:             %[[LenL:.*]] = arith.subi %[[P]], %[[L3]]
// CHECK:             %[[LenH:.*]] = arith.subi %[[H3]], %[[P]]
// CHECK:             %[[Cmp:.*]] = arith.cmpi ule, %[[LenL]], %[[LenH]]
// CHECK:             %[[L4:.*]] = arith.select %[[Cmp]], %[[PP1]], %[[L3]]
// CHECK:             %[[H4:.*]] = arith.select %[[Cmp]], %[[H3]], %[[P]]
// CHECK:             scf.if %[[Cmp]]
// CHECK:               func.call @_sparse_qsort_1_i8_f32_index(%[[L3]], %[[P]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:             else
// CHECK:               func.call @_sparse_qsort_1_i8_f32_index(%[[PP1]], %[[H3]], %[[X0]], %[[Y0]], %[[Y1]])
// CHECK:             scf.yield %[[L4]], %[[H4]]
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @sparse_sort_1d2v_quick(
// CHECK-SAME:                                %[[N:.*]]: index,
// CHECK-SAME:                                %[[X0:.*]]: memref<10xi8>,
// CHECK-SAME:                                %[[Y0:.*]]: memref<?xf32>,
// CHECK-SAME:                                %[[Y1:.*]]: memref<10xindex>) -> (memref<10xi8>, memref<?xf32>, memref<10xindex>) {
// CHECK:           %[[C0:.*]] = arith.constant 0
// CHECK:           %[[DX0:.*]] = memref.cast %[[X0]] : memref<10xi8> to memref<?xi8>
// CHECK:           %[[DY1:.*]] = memref.cast %[[Y1]] : memref<10xindex> to memref<?xindex>
// CHECK:           call @_sparse_qsort_1_i8_f32_index(%[[C0]], %[[N]], %[[DX0]], %[[Y0]], %[[DY1]])
// CHECK:           return %[[X0]], %[[Y0]], %[[Y1]]
// CHECK:         }
func.func @sparse_sort_1d2v_quick(%arg0: index, %arg1: memref<10xi8>, %arg2: memref<?xf32>, %arg3: memref<10xindex>)
   -> (memref<10xi8>, memref<?xf32>, memref<10xindex>) {
  sparse_tensor.sort quick_sort %arg0, %arg1 jointly %arg2, %arg3 : memref<10xi8> jointly memref<?xf32>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xi8>, memref<?xf32>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_qsort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_quick
func.func @sparse_sort_3d_quick(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort quick_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting function now. We have integration test
// to verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-DAG:     func.func private @_sparse_shift_down_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-DAG:     func.func private @_sparse_partition_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_hybrid_qsort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: i64) {
// CHECK-LABEL:   func.func @sparse_sort_3d_hybrid
func.func @sparse_sort_3d_hybrid(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort hybrid_quick_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_stable
func.func @sparse_sort_3d_stable(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort insertion_sort_stable %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_shift_down_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_3_index(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xindex>, %arg4: memref<?xindex>) {
// CHECK-LABEL:   func.func @sparse_sort_3d_heap
func.func @sparse_sort_3d_heap(%arg0: index, %arg1: memref<10xindex>, %arg2: memref<?xindex>, %arg3: memref<10xindex>) -> (memref<10xindex>, memref<?xindex>, memref<10xindex>) {
  sparse_tensor.sort heap_sort %arg0, %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
  return %arg1, %arg2, %arg3 : memref<10xindex>, memref<?xindex>, memref<10xindex>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_partition_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_qsort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_quick
func.func @sparse_sort_coo_quick(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo quick_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-DAG:     func.func private @_sparse_shift_down_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-DAG:     func.func private @_sparse_partition_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_hybrid_qsort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: i64) {
// CHECK-LABEL:   func.func @sparse_sort_coo_hybrid
func.func @sparse_sort_coo_hybrid(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo hybrid_quick_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_binary_search_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) -> index {
// CHECK-DAG:     func.func private @_sparse_sort_stable_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_stable
func.func @sparse_sort_coo_stable(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo insertion_sort_stable %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}

// -----

// Only check the generated supporting functions. We have integration test to
// verify correctness of the generated code.
//
// CHECK-DAG:     func.func private @_sparse_shift_down_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: index) {
// CHECK-DAG:     func.func private @_sparse_heap_sort_2_index_coo_1_f32_i32(%arg0: index, %arg1: index, %arg2: memref<?xindex>, %arg3: memref<?xf32>, %arg4: memref<?xi32>) {
// CHECK-LABEL:   func.func @sparse_sort_coo_heap
func.func @sparse_sort_coo_heap(%arg0: index, %arg1: memref<100xindex>, %arg2: memref<?xf32>, %arg3: memref<10xi32>) -> (memref<100xindex>, memref<?xf32>, memref<10xi32>) {
  sparse_tensor.sort_coo heap_sort %arg0, %arg1 jointly %arg2, %arg3 {nx = 2 : index, ny = 1: index} : memref<100xindex> jointly memref<?xf32>, memref<10xi32>
  return %arg1, %arg2, %arg3 : memref<100xindex>, memref<?xf32>, memref<10xi32>
}