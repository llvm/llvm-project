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
// CHECK-SAME:                                                        %[[VAL_0:.*0]]: index,
// CHECK-SAME:                                                        %[[VAL_1:.*1]]: index,
// CHECK-SAME:                                                        %[[VAL_2:.*2]]: memref<?xi8>,
// CHECK-SAME:                                                        %[[VAL_3:.*3]]: memref<?xf32>,
// CHECK-SAME:                                                        %[[VAL_4:.*4]]: memref<?xindex>) -> index {
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1000
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant -1
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]]
// CHECK:           %[[VAL_9:.*]] = arith.shrui %[[VAL_8]], %[[VAL_5]]
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_1]], %[[VAL_5]]
// CHECK:           %[[VAL_11:.*]] = arith.subi %[[VAL_1]], %[[VAL_0]]
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_6]]
// CHECK:           scf.if %[[VAL_12]] {
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_13]], %[[VAL_14]]
// CHECK:             scf.if %[[VAL_15]] {
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_17:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_17]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_16]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_19]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_18]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_21]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_20]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:             }
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_24:.*]] = arith.cmpi ult, %[[VAL_22]], %[[VAL_23]]
// CHECK:             scf.if %[[VAL_24]] {
// CHECK:               %[[VAL_25:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_26]], %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_25]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_27:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_28:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_28]], %[[VAL_3]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_27]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_29:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_30:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_30]], %[[VAL_4]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_29]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_31:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_32:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_33:.*]] = arith.cmpi ult, %[[VAL_31]], %[[VAL_32]]
// CHECK:               scf.if %[[VAL_33]] {
// CHECK:                 %[[VAL_34:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_35:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_35]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_34]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                 %[[VAL_36:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_37:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_37]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_36]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                 %[[VAL_38:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_39:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_39]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_38]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:               }
// CHECK:             }
// CHECK:           } else {
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]]
// CHECK:             %[[VAL_41:.*]] = arith.shrui %[[VAL_40]], %[[VAL_5]]
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_43:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:             %[[VAL_44:.*]] = arith.cmpi ult, %[[VAL_42]], %[[VAL_43]]
// CHECK:             scf.if %[[VAL_44]] {
// CHECK:               %[[VAL_45:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_46:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_46]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_45]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_47:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_48:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_48]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_47]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_49:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_50:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:               memref.store %[[VAL_50]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_49]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_52:.*]] = arith.cmpi ult, %[[VAL_51]], %[[VAL_51]]
// CHECK:             scf.if %[[VAL_52]] {
// CHECK:               %[[VAL_53:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_53]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_53]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_54:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_54]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_54]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_55:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_55]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_55]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_56:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_57:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:               %[[VAL_58:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_57]]
// CHECK:               scf.if %[[VAL_58]] {
// CHECK:                 %[[VAL_59:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_60:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_60]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_59]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                 %[[VAL_61:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_62:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_62]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_61]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                 %[[VAL_63:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_64:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                 memref.store %[[VAL_64]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_63]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:               }
// CHECK:             }
// CHECK:             %[[VAL_65:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:             %[[VAL_66:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:             %[[VAL_67:.*]] = arith.cmpi ult, %[[VAL_65]], %[[VAL_66]]
// CHECK:             scf.if %[[VAL_67]] {
// CHECK:               %[[VAL_68:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_69:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_69]], %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_68]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_70:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_71:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_71]], %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_70]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_72:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_73:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               memref.store %[[VAL_73]], %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_72]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_74:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_75:.*]] = arith.cmpi ult, %[[VAL_74]], %[[VAL_74]]
// CHECK:               scf.if %[[VAL_75]] {
// CHECK:                 %[[VAL_76:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_76]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_76]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_77:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_77]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_77]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_78:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_78]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_78]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_79:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_80:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                 %[[VAL_81:.*]] = arith.cmpi ult, %[[VAL_79]], %[[VAL_80]]
// CHECK:                 scf.if %[[VAL_81]] {
// CHECK:                   %[[VAL_82:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_83:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                   memref.store %[[VAL_83]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_82]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                   %[[VAL_84:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_85:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                   memref.store %[[VAL_85]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_84]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                   %[[VAL_86:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_87:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                   memref.store %[[VAL_87]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_86]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             %[[VAL_88:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:             %[[VAL_89:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:             %[[VAL_90:.*]] = arith.cmpi ult, %[[VAL_88]], %[[VAL_89]]
// CHECK:             scf.if %[[VAL_90]] {
// CHECK:               %[[VAL_91:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_92:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_92]], %[[VAL_2]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_91]], %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_93:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_94:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_94]], %[[VAL_3]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_93]], %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_95:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_10]]]
// CHECK:               %[[VAL_96:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:               memref.store %[[VAL_96]], %[[VAL_4]]{{\[}}%[[VAL_10]]]
// CHECK:               memref.store %[[VAL_95]], %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_97:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:               %[[VAL_98:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:               %[[VAL_99:.*]] = arith.cmpi ult, %[[VAL_97]], %[[VAL_98]]
// CHECK:               scf.if %[[VAL_99]] {
// CHECK:                 %[[VAL_100:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:                 %[[VAL_101:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_101]], %[[VAL_2]]{{\[}}%[[VAL_41]]]
// CHECK:                 memref.store %[[VAL_100]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_102:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:                 %[[VAL_103:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_103]], %[[VAL_3]]{{\[}}%[[VAL_41]]]
// CHECK:                 memref.store %[[VAL_102]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_104:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:                 %[[VAL_105:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 memref.store %[[VAL_105]], %[[VAL_4]]{{\[}}%[[VAL_41]]]
// CHECK:                 memref.store %[[VAL_104]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_106:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                 %[[VAL_107:.*]] = arith.cmpi ult, %[[VAL_106]], %[[VAL_106]]
// CHECK:                 scf.if %[[VAL_107]] {
// CHECK:                   %[[VAL_108:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_108]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_108]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_109:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_109]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_109]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_110:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_110]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                   memref.store %[[VAL_110]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_111:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                   %[[VAL_112:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                   %[[VAL_113:.*]] = arith.cmpi ult, %[[VAL_111]], %[[VAL_112]]
// CHECK:                   scf.if %[[VAL_113]] {
// CHECK:                     %[[VAL_114:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                     %[[VAL_115:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                     memref.store %[[VAL_115]], %[[VAL_2]]{{\[}}%[[VAL_9]]]
// CHECK:                     memref.store %[[VAL_114]], %[[VAL_2]]{{\[}}%[[VAL_0]]]
// CHECK:                     %[[VAL_116:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                     %[[VAL_117:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                     memref.store %[[VAL_117]], %[[VAL_3]]{{\[}}%[[VAL_9]]]
// CHECK:                     memref.store %[[VAL_116]], %[[VAL_3]]{{\[}}%[[VAL_0]]]
// CHECK:                     %[[VAL_118:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                     %[[VAL_119:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                     memref.store %[[VAL_119]], %[[VAL_4]]{{\[}}%[[VAL_9]]]
// CHECK:                     memref.store %[[VAL_118]], %[[VAL_4]]{{\[}}%[[VAL_0]]]
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_120:.*]]:3 = scf.while (%[[VAL_121:.*]] = %[[VAL_0]], %[[VAL_122:.*]] = %[[VAL_10]], %[[VAL_123:.*]] = %[[VAL_9]])
// CHECK:             %[[VAL_124:.*]] = arith.cmpi ult, %[[VAL_121]], %[[VAL_122]]
// CHECK:             scf.condition(%[[VAL_124]]) %[[VAL_121]], %[[VAL_122]], %[[VAL_123]]
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_125:.*]]: index, %[[VAL_126:.*]]: index, %[[VAL_127:.*]]: index)
// CHECK:             %[[VAL_128:.*]] = scf.while (%[[VAL_129:.*]] = %[[VAL_125]])
// CHECK:               %[[VAL_130:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_129]]]
// CHECK:               %[[VAL_131:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_127]]]
// CHECK:               %[[VAL_132:.*]] = arith.cmpi ult, %[[VAL_130]], %[[VAL_131]]
// CHECK:               scf.condition(%[[VAL_132]]) %[[VAL_129]]
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_133:.*]]: index):
// CHECK:               %[[VAL_134:.*]] = arith.addi %[[VAL_133]], %[[VAL_5]]
// CHECK:               scf.yield %[[VAL_134]]
// CHECK:             }
// CHECK:             %[[VAL_135:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_136:.*]]]
// CHECK:             %[[VAL_137:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_127]]]
// CHECK:             %[[VAL_138:.*]] = arith.cmpi eq, %[[VAL_135]], %[[VAL_137]]
// CHECK:             %[[VAL_139:.*]] = scf.while (%[[VAL_140:.*]] = %[[VAL_126]])
// CHECK:               %[[VAL_141:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_127]]]
// CHECK:               %[[VAL_142:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_140]]]
// CHECK:               %[[VAL_143:.*]] = arith.cmpi ult, %[[VAL_141]], %[[VAL_142]]
// CHECK:               scf.condition(%[[VAL_143]]) %[[VAL_140]]
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_144:.*]]: index):
// CHECK:               %[[VAL_145:.*]] = arith.addi %[[VAL_144]], %[[VAL_7]]
// CHECK:               scf.yield %[[VAL_145]]
// CHECK:             }
// CHECK:             %[[VAL_146:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_147:.*]]]
// CHECK:             %[[VAL_148:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_127]]]
// CHECK:             %[[VAL_149:.*]] = arith.cmpi eq, %[[VAL_146]], %[[VAL_148]]
// CHECK:             %[[VAL_150:.*]] = arith.cmpi ult, %[[VAL_136]], %[[VAL_147]]
// CHECK:             %[[VAL_151:.*]]:3 = scf.if %[[VAL_150]]
// CHECK:               %[[VAL_152:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_136]]]
// CHECK:               %[[VAL_153:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_147]]]
// CHECK:               memref.store %[[VAL_153]], %[[VAL_2]]{{\[}}%[[VAL_136]]]
// CHECK:               memref.store %[[VAL_152]], %[[VAL_2]]{{\[}}%[[VAL_147]]]
// CHECK:               %[[VAL_154:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_136]]]
// CHECK:               %[[VAL_155:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_147]]]
// CHECK:               memref.store %[[VAL_155]], %[[VAL_3]]{{\[}}%[[VAL_136]]]
// CHECK:               memref.store %[[VAL_154]], %[[VAL_3]]{{\[}}%[[VAL_147]]]
// CHECK:               %[[VAL_156:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_136]]]
// CHECK:               %[[VAL_157:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_147]]]
// CHECK:               memref.store %[[VAL_157]], %[[VAL_4]]{{\[}}%[[VAL_136]]]
// CHECK:               memref.store %[[VAL_156]], %[[VAL_4]]{{\[}}%[[VAL_147]]]
// CHECK:               %[[VAL_158:.*]] = arith.cmpi eq, %[[VAL_136]], %[[VAL_127]]
// CHECK:               %[[VAL_159:.*]] = scf.if %[[VAL_158]]
// CHECK:                 scf.yield %[[VAL_147]]
// CHECK:               } else {
// CHECK:                 %[[VAL_160:.*]] = arith.cmpi eq, %[[VAL_147]], %[[VAL_127]]
// CHECK:                 %[[VAL_161:.*]] = arith.select %[[VAL_160]], %[[VAL_136]], %[[VAL_127]]
// CHECK:                 scf.yield %[[VAL_161]]
// CHECK:               }
// CHECK:               %[[VAL_162:.*]] = arith.andi %[[VAL_138]], %[[VAL_149]] : i1
// CHECK:               %[[VAL_163:.*]]:2 = scf.if %[[VAL_162]]
// CHECK:                 %[[VAL_164:.*]] = arith.addi %[[VAL_136]], %[[VAL_5]]
// CHECK:                 %[[VAL_165:.*]] = arith.subi %[[VAL_147]], %[[VAL_5]]
// CHECK:                 scf.yield %[[VAL_164]], %[[VAL_165]]
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_136]], %[[VAL_147]]
// CHECK:               }
// CHECK:               scf.yield %[[VAL_166:.*]]#0, %[[VAL_166]]#1, %[[VAL_167:.*]]
// CHECK:             } else {
// CHECK:               scf.yield %[[VAL_136]], %[[VAL_147]], %[[VAL_127]]
// CHECK:             }
// CHECK:             scf.yield %[[VAL_168:.*]]#0, %[[VAL_168]]#1, %[[VAL_168]]#2
// CHECK:           }
// CHECK:           return %[[VAL_169:.*]]#2
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