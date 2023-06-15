// RUN: mlir-opt %s -sparsification --canonicalize | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

#trait_scale = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = X(i,j) * 2.0"
}

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel","reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}

#trait_mul = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // X
    affine_map<(i,j) -> (i,j)>, // Y
    affine_map<(i,j) -> (i,j)>  // Z (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "Z(i,j) = X(i,j) * Y(i,j)"
}

//
// Kernels that operate on SortedCOO format.
//

// CHECK-LABEL:   func.func @sparse_scale(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xf32>
// CHECK-DAG:       %[[VAL_8:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-DAG:       %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = scf.while (%[[VAL_11:.*]] = %[[VAL_8]]) : (index) -> index {
// CHECK:             %[[VAL_12:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_9]] : index
// CHECK:             scf.condition(%[[VAL_12]]) %[[VAL_11]] : index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*]]: index):
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_15:.*]] = scf.while (%[[VAL_16:.*]] = %[[VAL_13]]) : (index) -> index {
// CHECK:               %[[VAL_17:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_9]] : index
// CHECK:               %[[VAL_18:.*]] = scf.if %[[VAL_17]] -> (i1) {
// CHECK:                 %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_14]] : index
// CHECK:                 scf.yield %[[VAL_20]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_1]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_21:.*]]) %[[VAL_16]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_22:.*]]: index):
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// CHECK:               scf.yield %[[VAL_23]] : index
// CHECK:             }
// CHECK:             scf.for %[[VAL_24:.*]] = %[[VAL_13]] to %[[VAL_25:.*]] step %[[VAL_3]] {
// CHECK:               %[[VAL_26:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK:               %[[VAL_27:.*]] = arith.mulf %[[VAL_26]], %[[VAL_4]] : f32
// CHECK:               memref.store %[[VAL_27]], %[[VAL_7]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             scf.yield %[[VAL_28:.*]] : index
// CHECK:           } attributes {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_29:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>
// CHECK:           return %[[VAL_29]] : tensor<?x?xf32, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>
// CHECK:         }
func.func @sparse_scale(%argx: tensor<?x?xf32, #SortedCOO>) -> tensor<?x?xf32, #SortedCOO> {
  %c = arith.constant 2.0 : f32
  %0 = linalg.generic #trait_scale
    outs(%argx: tensor<?x?xf32, #SortedCOO>) {
      ^bb(%x: f32):
        %1 = arith.mulf %x, %c : f32
        linalg.yield %1 : f32
  } -> tensor<?x?xf32, #SortedCOO>
  return %0 : tensor<?x?xf32, #SortedCOO>
}

// CHECK-LABEL:   func.func @matvec(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = scf.while (%[[VAL_14:.*]] = %[[VAL_11]]) : (index) -> index {
// CHECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_12]] : index
// CHECK:             scf.condition(%[[VAL_15]]) %[[VAL_14]] : index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_16:.*]]: index):
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_19:.*]] = scf.while (%[[VAL_20:.*]] = %[[VAL_16]]) : (index) -> index {
// CHECK:               %[[VAL_21:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_12]] : index
// CHECK:               %[[VAL_22:.*]] = scf.if %[[VAL_21]] -> (i1) {
// CHECK:                 %[[VAL_23:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_20]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_24:.*]] = arith.cmpi eq, %[[VAL_23]], %[[VAL_18]] : index
// CHECK:                 scf.yield %[[VAL_24]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_3]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_25:.*]]) %[[VAL_20]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_26:.*]]: index):
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_5]] : index
// CHECK:               scf.yield %[[VAL_27]] : index
// CHECK:             }
// CHECK:             %[[VAL_28:.*]] = tensor.extract %[[VAL_2]]{{\[}}%[[VAL_17]]] : tensor<32xf64>
// CHECK:             %[[VAL_29:.*]] = scf.for %[[VAL_30:.*]] = %[[VAL_16]] to %[[VAL_31:.*]] step %[[VAL_5]] iter_args(%[[VAL_32:.*]] = %[[VAL_28]]) -> (f64) {
// CHECK:               %[[VAL_33:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_30]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_30]]] : memref<?xf64>
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_33]]] : tensor<64xf64>
// CHECK:               %[[VAL_36:.*]] = arith.mulf %[[VAL_34]], %[[VAL_35]] : f64
// CHECK:               %[[VAL_37:.*]] = arith.addf %[[VAL_32]], %[[VAL_36]] : f64
// CHECK:               scf.yield %[[VAL_37]] : f64
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             memref.store %[[VAL_38:.*]], %[[VAL_10]]{{\[}}%[[VAL_17]]] : memref<32xf64>
// CHECK:             scf.yield %[[VAL_39:.*]] : index
// CHECK:           } attributes {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_40:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<32xf64>
// CHECK:           return %[[VAL_40]] : tensor<32xf64>
// CHECK:         }
func.func @matvec(%arga: tensor<32x64xf64, #SortedCOO>,
                  %argb: tensor<64xf64>,
                  %argx: tensor<32xf64>) -> tensor<32xf64> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<32x64xf64, #SortedCOO>, tensor<64xf64>)
      outs(%argx: tensor<32xf64>) {
    ^bb(%A: f64, %b: f64, %x: f64):
      %0 = arith.mulf %A, %b : f64
      %1 = arith.addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}

// CHECK-LABEL:   func.func @mateltmul(
// CHECK-SAME:      %[[VAL_0:.*0]]: tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:      %[[VAL_1:.*1]]: tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:      %[[VAL_2:.*2]]: tensor<32x64xf64>) -> tensor<32x64xf64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_15:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32x64xf64>
// CHECK:           linalg.fill ins(%[[VAL_4]] : f64) outs(%[[VAL_15]] : memref<32x64xf64>)
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK:           %[[VAL_20:.*]]:2 = scf.while (%[[VAL_21:.*]] = %[[VAL_16]], %[[VAL_22:.*]] = %[[VAL_18]]) : (index, index) -> (index, index) {
// CHECK:             %[[VAL_23:.*]] = arith.cmpi ult, %[[VAL_21]], %[[VAL_17]] : index
// CHECK:             %[[VAL_24:.*]] = arith.cmpi ult, %[[VAL_22]], %[[VAL_19]] : index
// CHECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i1
// CHECK:             scf.condition(%[[VAL_25]]) %[[VAL_21]], %[[VAL_22]] : index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index):
// CHECK:             %[[VAL_28:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_26]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_29:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_27]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_32:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_26]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_33:.*]] = scf.while (%[[VAL_34:.*]] = %[[VAL_26]]) : (index) -> index {
// CHECK:               %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_34]], %[[VAL_17]] : index
// CHECK:               %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i1) {
// CHECK:                 %[[VAL_37:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_34]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_37]], %[[VAL_32]] : index
// CHECK:                 scf.yield %[[VAL_38]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_3]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_39:.*]]) %[[VAL_34]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_40:.*]]: index):
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_6]] : index
// CHECK:               scf.yield %[[VAL_41]] : index
// CHECK:             }
// CHECK:             %[[VAL_42:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_27]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:             %[[VAL_43:.*]] = scf.while (%[[VAL_44:.*]] = %[[VAL_27]]) : (index) -> index {
// CHECK:               %[[VAL_45:.*]] = arith.cmpi ult, %[[VAL_44]], %[[VAL_19]] : index
// CHECK:               %[[VAL_46:.*]] = scf.if %[[VAL_45]] -> (i1) {
// CHECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_44]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_47]], %[[VAL_42]] : index
// CHECK:                 scf.yield %[[VAL_48]] : i1
// CHECK:               } else {
// CHECK:                 scf.yield %[[VAL_3]] : i1
// CHECK:               }
// CHECK:               scf.condition(%[[VAL_49:.*]]) %[[VAL_44]] : index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_50:.*]]: index):
// CHECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_6]] : index
// CHECK:               scf.yield %[[VAL_51]] : index
// CHECK:             }
// CHECK:             %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_29]], %[[VAL_28]] : index
// CHECK:             %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_28]] : index
// CHECK:             %[[VAL_52:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_31]] : index
// CHECK:             %[[VAL_53:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_31]] : index
// CHECK:             %[[VAL_54:.*]] = arith.andi %[[VAL_52]], %[[VAL_53]] : i1
// CHECK:             scf.if %[[VAL_54]] {
// CHECK:               %[[VAL_55:.*]]:2 = scf.while (%[[VAL_56:.*]] = %[[VAL_26]], %[[VAL_57:.*]] = %[[VAL_27]]) : (index, index) -> (index, index) {
// CHECK:                 %[[VAL_58:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_59:.*]] : index
// CHECK:                 %[[VAL_60:.*]] = arith.cmpi ult, %[[VAL_57]], %[[VAL_61:.*]] : index
// CHECK:                 %[[VAL_62:.*]] = arith.andi %[[VAL_58]], %[[VAL_60]] : i1
// CHECK:                 scf.condition(%[[VAL_62]]) %[[VAL_56]], %[[VAL_57]] : index, index
// CHECK:               } do {
// CHECK:               ^bb0(%[[VAL_63:.*]]: index, %[[VAL_64:.*]]: index):
// CHECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_63]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_64]]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:                 %[[VAL_67:.*]] = arith.cmpi ult, %[[VAL_66]], %[[VAL_65]] : index
// CHECK:                 %[[VAL_68:.*]] = arith.select %[[VAL_67]], %[[VAL_66]], %[[VAL_65]] : index
// CHECK:                 %[[VAL_69:.*]] = arith.cmpi eq, %[[VAL_65]], %[[VAL_68]] : index
// CHECK:                 %[[VAL_70:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_68]] : index
// CHECK:                 %[[VAL_71:.*]] = arith.andi %[[VAL_69]], %[[VAL_70]] : i1
// CHECK:                 scf.if %[[VAL_71]] {
// CHECK:                   %[[VAL_72:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_63]]] : memref<?xf64>
// CHECK:                   %[[VAL_73:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_64]]] : memref<?xf64>
// CHECK:                   %[[VAL_74:.*]] = arith.mulf %[[VAL_72]], %[[VAL_73]] : f64
// CHECK:                   memref.store %[[VAL_74]], %[[VAL_15]]{{\[}}%[[VAL_31]], %[[VAL_68]]] : memref<32x64xf64>
// CHECK:                 }
// CHECK:                 %[[VAL_75:.*]] = arith.cmpi eq, %[[VAL_65]], %[[VAL_68]] : index
// CHECK:                 %[[VAL_76:.*]] = arith.addi %[[VAL_63]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_77:.*]] = arith.select %[[VAL_75]], %[[VAL_76]], %[[VAL_63]] : index
// CHECK:                 %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_68]] : index
// CHECK:                 %[[VAL_79:.*]] = arith.addi %[[VAL_64]], %[[VAL_6]] : index
// CHECK:                 %[[VAL_80:.*]] = arith.select %[[VAL_78]], %[[VAL_79]], %[[VAL_64]] : index
// CHECK:                 scf.yield %[[VAL_77]], %[[VAL_80]] : index, index
// CHECK:               } attributes {"Emitted from" = "linalg.generic"}
// CHECK:             }
// CHECK:             %[[VAL_81:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_31]] : index
// CHECK:             %[[VAL_82:.*]] = arith.select %[[VAL_81]], %[[VAL_83:.*]], %[[VAL_26]] : index
// CHECK:             %[[VAL_84:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_31]] : index
// CHECK:             %[[VAL_85:.*]] = arith.select %[[VAL_84]], %[[VAL_86:.*]], %[[VAL_27]] : index
// CHECK:             scf.yield %[[VAL_82]], %[[VAL_85]] : index, index
// CHECK:           } attributes {"Emitted from" = "linalg.generic"}
// CHECK:           %[[VAL_87:.*]] = bufferization.to_tensor %[[VAL_15]] : memref<32x64xf64>
// CHECK:           return %[[VAL_87]] : tensor<32x64xf64>
// CHECK:         }
func.func @mateltmul(%argx: tensor<32x64xf64, #SortedCOO>,
                     %argy: tensor<32x64xf64, #SortedCOO>,
                     %argz: tensor<32x64xf64>) -> tensor<32x64xf64> {
  %0 = linalg.generic #trait_mul
      ins(%argx, %argy : tensor<32x64xf64, #SortedCOO>, tensor<32x64xf64, #SortedCOO>)
      outs(%argz: tensor<32x64xf64>) {
    ^bb(%x: f64, %y: f64, %z: f64):
      %1 = arith.mulf %x, %y : f64
      linalg.yield %1 : f64
  } -> tensor<32x64xf64>
  return %0 : tensor<32x64xf64>
}
