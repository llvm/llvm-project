// TODO: re-enable after lowering coo.next to function call (such that loop structure is more clear).
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification --canonicalize | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
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
// C_HECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32, #sparse{{[0-9]*}}>) -> tensor<?x?xf32, #sparse{{[0-9]*}}> {
// C_HECK-DAG:       %[[VAL_1:.*]] = arith.constant false
// C_HECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_4:.*]] = arith.constant 2.000000e+00 : f32
// C_HECK-DAG:       %[[VAL_5:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// C_HECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
// C_HECK-DAG:       %[[VAL_8:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// C_HECK-DAG:       %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// C_HECK:           %[[VAL_10:.*]] = scf.while (%[[VAL_11:.*]] = %[[VAL_8]]) : (index) -> index {
// C_HECK:             %[[VAL_12:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_9]] : index
// C_HECK:             scf.condition(%[[VAL_12]]) %[[VAL_11]] : index
// C_HECK:           } do {
// C_HECK:           ^bb0(%[[VAL_13:.*]]: index):
// C_HECK:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_15:.*]] = scf.while (%[[VAL_16:.*]] = %[[VAL_13]]) : (index) -> index {
// C_HECK:               %[[VAL_17:.*]] = arith.cmpi ult, %[[VAL_16]], %[[VAL_9]] : index
// C_HECK:               %[[VAL_18:.*]] = scf.if %[[VAL_17]] -> (i1) {
// C_HECK:                 %[[VAL_19:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_19]], %[[VAL_14]] : index
// C_HECK:                 scf.yield %[[VAL_20]] : i1
// C_HECK:               } else {
// C_HECK:                 scf.yield %[[VAL_1]] : i1
// C_HECK:               }
// C_HECK:               scf.condition(%[[VAL_21:.*]]) %[[VAL_16]] : index
// C_HECK:             } do {
// C_HECK:             ^bb0(%[[VAL_22:.*]]: index):
// C_HECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : index
// C_HECK:               scf.yield %[[VAL_23]] : index
// C_HECK:             }
// C_HECK:             scf.for %[[VAL_24:.*]] = %[[VAL_13]] to %[[VAL_25:.*]] step %[[VAL_3]] {
// C_HECK:               %[[VAL_26:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// C_HECK:               %[[VAL_27:.*]] = arith.mulf %[[VAL_26]], %[[VAL_4]] : f32
// C_HECK:               memref.store %[[VAL_27]], %[[VAL_7]]{{\[}}%[[VAL_24]]] : memref<?xf32>
// C_HECK:             } {"Emitted from" = "linalg.generic"}
// C_HECK:             scf.yield %[[VAL_28:.*]] : index
// C_HECK:           } attributes {"Emitted from" = "linalg.generic"}
// C_HECK:           %[[VAL_29:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// C_HECK:           return %[[VAL_29]] : tensor<?x?xf32, #sparse{{[0-9]*}}>
// C_HECK:         }
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

// C_HECK-LABEL:   func.func @matvec(
// C_HECK-SAME:      %[[VAL_0:.*]]: tensor<32x64xf64, #sparse{{[0-9]*}}>,
// C_HECK-SAME:      %[[VAL_1:.*]]: tensor<64xf64>,
// C_HECK-SAME:      %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// C_HECK-DAG:       %[[VAL_3:.*]] = arith.constant false
// C_HECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// C_HECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// C_HECK:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : tensor<32xf64> to memref<32xf64>
// C_HECK:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// C_HECK:           %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// C_HECK:           %[[VAL_13:.*]] = scf.while (%[[VAL_14:.*]] = %[[VAL_11]]) : (index) -> index {
// C_HECK:             %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_12]] : index
// C_HECK:             scf.condition(%[[VAL_15]]) %[[VAL_14]] : index
// C_HECK:           } do {
// C_HECK:           ^bb0(%[[VAL_16:.*]]: index):
// C_HECK:             %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_18:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_16]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_19:.*]] = scf.while (%[[VAL_20:.*]] = %[[VAL_16]]) : (index) -> index {
// C_HECK:               %[[VAL_21:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_12]] : index
// C_HECK:               %[[VAL_22:.*]] = scf.if %[[VAL_21]] -> (i1) {
// C_HECK:                 %[[VAL_23:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_20]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_24:.*]] = arith.cmpi eq, %[[VAL_23]], %[[VAL_18]] : index
// C_HECK:                 scf.yield %[[VAL_24]] : i1
// C_HECK:               } else {
// C_HECK:                 scf.yield %[[VAL_3]] : i1
// C_HECK:               }
// C_HECK:               scf.condition(%[[VAL_25:.*]]) %[[VAL_20]] : index
// C_HECK:             } do {
// C_HECK:             ^bb0(%[[VAL_26:.*]]: index):
// C_HECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_5]] : index
// C_HECK:               scf.yield %[[VAL_27]] : index
// C_HECK:             }
// C_HECK:             %[[VAL_28:.*]] = tensor.extract %[[VAL_2]]{{\[}}%[[VAL_17]]] : tensor<32xf64>
// C_HECK:             %[[VAL_29:.*]] = scf.for %[[VAL_30:.*]] = %[[VAL_16]] to %[[VAL_31:.*]] step %[[VAL_5]] iter_args(%[[VAL_32:.*]] = %[[VAL_28]]) -> (f64) {
// C_HECK:               %[[VAL_33:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_30]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:               %[[VAL_34:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_30]]] : memref<?xf64>
// C_HECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_1]]{{\[}}%[[VAL_33]]] : tensor<64xf64>
// C_HECK:               %[[VAL_36:.*]] = arith.mulf %[[VAL_34]], %[[VAL_35]] : f64
// C_HECK:               %[[VAL_37:.*]] = arith.addf %[[VAL_32]], %[[VAL_36]] : f64
// C_HECK:               scf.yield %[[VAL_37]] : f64
// C_HECK:             } {"Emitted from" = "linalg.generic"}
// C_HECK:             memref.store %[[VAL_38:.*]], %[[VAL_10]]{{\[}}%[[VAL_17]]] : memref<32xf64>
// C_HECK:             scf.yield %[[VAL_39:.*]] : index
// C_HECK:           } attributes {"Emitted from" = "linalg.generic"}
// C_HECK:           %[[VAL_40:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<32xf64>
// C_HECK:           return %[[VAL_40]] : tensor<32xf64>
// C_HECK:         }
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

// C_HECK-LABEL:   func.func @mateltmul(
// C_HECK-SAME:      %[[VAL_0:.*0]]: tensor<32x64xf64, #sparse{{[0-9]*}}>, %[[VAL_1:.*1]]: tensor<32x64xf64, #sparse{{[0-9]*}}>,
// C_HECK-SAME:      %[[VAL_2:.*2]]: tensor<32x64xf64>) -> tensor<32x64xf64> {
// C_HECK-DAG:       %[[VAL_3:.*]] = arith.constant false
// C_HECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// C_HECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// C_HECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// C_HECK-DAG:       %[[VAL_7:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// C_HECK-DAG:       %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// C_HECK-DAG:       %[[VAL_11:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex>
// C_HECK-DAG:       %[[VAL_12:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 0 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_13:.*]] = sparse_tensor.coordinates %[[VAL_1]] {level = 1 : index} : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xindex, strided<[?], offset: ?>>
// C_HECK-DAG:       %[[VAL_14:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<32x64xf64, #sparse{{[0-9]*}}> to memref<?xf64>
// C_HECK:           %[[VAL_15:.*]] = bufferization.to_memref %[[VAL_2]] : tensor<32x64xf64> to memref<32x64xf64>
// C_HECK:           linalg.fill ins(%[[VAL_4]] : f64) outs(%[[VAL_15]] : memref<32x64xf64>)
// C_HECK:           %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// C_HECK:           %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// C_HECK:           %[[VAL_18:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// C_HECK:           %[[VAL_19:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// C_HECK:           %[[VAL_20:.*]]:2 = scf.while (%[[VAL_21:.*]] = %[[VAL_16]], %[[VAL_22:.*]] = %[[VAL_18]]) : (index, index) -> (index, index) {
// C_HECK:             %[[VAL_23:.*]] = arith.cmpi ult, %[[VAL_21]], %[[VAL_17]] : index
// C_HECK:             %[[VAL_24:.*]] = arith.cmpi ult, %[[VAL_22]], %[[VAL_19]] : index
// C_HECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i1
// C_HECK:             scf.condition(%[[VAL_25]]) %[[VAL_21]], %[[VAL_22]] : index, index
// C_HECK:           } do {
// C_HECK:           ^bb0(%[[VAL_26:.*]]: index, %[[VAL_27:.*]]: index):
// C_HECK:             %[[VAL_28:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_26]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_29:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_27]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_32:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_26]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_33:.*]] = scf.while (%[[VAL_34:.*]] = %[[VAL_26]]) : (index) -> index {
// C_HECK:               %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_34]], %[[VAL_17]] : index
// C_HECK:               %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i1) {
// C_HECK:                 %[[VAL_37:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_34]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_37]], %[[VAL_32]] : index
// C_HECK:                 scf.yield %[[VAL_38]] : i1
// C_HECK:               } else {
// C_HECK:                 scf.yield %[[VAL_3]] : i1
// C_HECK:               }
// C_HECK:               scf.condition(%[[VAL_39:.*]]) %[[VAL_34]] : index
// C_HECK:             } do {
// C_HECK:             ^bb0(%[[VAL_40:.*]]: index):
// C_HECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_6]] : index
// C_HECK:               scf.yield %[[VAL_41]] : index
// C_HECK:             }
// C_HECK:             %[[VAL_42:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_27]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:             %[[VAL_43:.*]] = scf.while (%[[VAL_44:.*]] = %[[VAL_27]]) : (index) -> index {
// C_HECK:               %[[VAL_45:.*]] = arith.cmpi ult, %[[VAL_44]], %[[VAL_19]] : index
// C_HECK:               %[[VAL_46:.*]] = scf.if %[[VAL_45]] -> (i1) {
// C_HECK:                 %[[VAL_47:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_44]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_47]], %[[VAL_42]] : index
// C_HECK:                 scf.yield %[[VAL_48]] : i1
// C_HECK:               } else {
// C_HECK:                 scf.yield %[[VAL_3]] : i1
// C_HECK:               }
// C_HECK:               scf.condition(%[[VAL_49:.*]]) %[[VAL_44]] : index
// C_HECK:             } do {
// C_HECK:             ^bb0(%[[VAL_50:.*]]: index):
// C_HECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_6]] : index
// C_HECK:               scf.yield %[[VAL_51]] : index
// C_HECK:             }
// C_HECK:             %[[VAL_30:.*]] = arith.cmpi ult, %[[VAL_29]], %[[VAL_28]] : index
// C_HECK:             %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_28]] : index
// C_HECK:             %[[VAL_52:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_31]] : index
// C_HECK:             %[[VAL_53:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_31]] : index
// C_HECK:             %[[VAL_54:.*]] = arith.andi %[[VAL_52]], %[[VAL_53]] : i1
// C_HECK:             scf.if %[[VAL_54]] {
// C_HECK:               %[[VAL_55:.*]]:2 = scf.while (%[[VAL_56:.*]] = %[[VAL_26]], %[[VAL_57:.*]] = %[[VAL_27]]) : (index, index) -> (index, index) {
// C_HECK:                 %[[VAL_58:.*]] = arith.cmpi ult, %[[VAL_56]], %[[VAL_59:.*]] : index
// C_HECK:                 %[[VAL_60:.*]] = arith.cmpi ult, %[[VAL_57]], %[[VAL_61:.*]] : index
// C_HECK:                 %[[VAL_62:.*]] = arith.andi %[[VAL_58]], %[[VAL_60]] : i1
// C_HECK:                 scf.condition(%[[VAL_62]]) %[[VAL_56]], %[[VAL_57]] : index, index
// C_HECK:               } do {
// C_HECK:               ^bb0(%[[VAL_63:.*]]: index, %[[VAL_64:.*]]: index):
// C_HECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_63]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_66:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_64]]] : memref<?xindex, strided<[?], offset: ?>>
// C_HECK:                 %[[VAL_67:.*]] = arith.cmpi ult, %[[VAL_66]], %[[VAL_65]] : index
// C_HECK:                 %[[VAL_68:.*]] = arith.select %[[VAL_67]], %[[VAL_66]], %[[VAL_65]] : index
// C_HECK:                 %[[VAL_69:.*]] = arith.cmpi eq, %[[VAL_65]], %[[VAL_68]] : index
// C_HECK:                 %[[VAL_70:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_68]] : index
// C_HECK:                 %[[VAL_71:.*]] = arith.andi %[[VAL_69]], %[[VAL_70]] : i1
// C_HECK:                 scf.if %[[VAL_71]] {
// C_HECK:                   %[[VAL_72:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_63]]] : memref<?xf64>
// C_HECK:                   %[[VAL_73:.*]] = memref.load %[[VAL_14]]{{\[}}%[[VAL_64]]] : memref<?xf64>
// C_HECK:                   %[[VAL_74:.*]] = arith.mulf %[[VAL_72]], %[[VAL_73]] : f64
// C_HECK:                   memref.store %[[VAL_74]], %[[VAL_15]]{{\[}}%[[VAL_31]], %[[VAL_68]]] : memref<32x64xf64>
// C_HECK:                 }
// C_HECK:                 %[[VAL_75:.*]] = arith.cmpi eq, %[[VAL_65]], %[[VAL_68]] : index
// C_HECK:                 %[[VAL_76:.*]] = arith.addi %[[VAL_63]], %[[VAL_6]] : index
// C_HECK:                 %[[VAL_77:.*]] = arith.select %[[VAL_75]], %[[VAL_76]], %[[VAL_63]] : index
// C_HECK:                 %[[VAL_78:.*]] = arith.cmpi eq, %[[VAL_66]], %[[VAL_68]] : index
// C_HECK:                 %[[VAL_79:.*]] = arith.addi %[[VAL_64]], %[[VAL_6]] : index
// C_HECK:                 %[[VAL_80:.*]] = arith.select %[[VAL_78]], %[[VAL_79]], %[[VAL_64]] : index
// C_HECK:                 scf.yield %[[VAL_77]], %[[VAL_80]] : index, index
// C_HECK:               } attributes {"Emitted from" = "linalg.generic"}
// C_HECK:             }
// C_HECK:             %[[VAL_81:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_31]] : index
// C_HECK:             %[[VAL_82:.*]] = arith.select %[[VAL_81]], %[[VAL_83:.*]], %[[VAL_26]] : index
// C_HECK:             %[[VAL_84:.*]] = arith.cmpi eq, %[[VAL_29]], %[[VAL_31]] : index
// C_HECK:             %[[VAL_85:.*]] = arith.select %[[VAL_84]], %[[VAL_86:.*]], %[[VAL_27]] : index
// C_HECK:             scf.yield %[[VAL_82]], %[[VAL_85]] : index, index
// C_HECK:           } attributes {"Emitted from" = "linalg.generic"}
// C_HECK:           %[[VAL_87:.*]] = bufferization.to_tensor %[[VAL_15]] : memref<32x64xf64>
// C_HECK:           return %[[VAL_87]] : tensor<32x64xf64>
// C_HECK:         }
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
