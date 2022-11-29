// RUN: mlir-opt %s -sparsification -cse -sparse-vectorization="vl=8" -cse | \
// RUN:   FileCheck %s

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a (in)
    affine_map<(i,j) -> (i,j)>,  // b (in)
    affine_map<(i,j) -> ()>      // x (out)
  ],
  iterator_types = ["reduction", "reduction"]
}

//
// Verifies that the SIMD reductions in the two for-loops after the
// while-loop are chained before horizontally reducing these back to scalar.
//
// CHECK-LABEL:   func.func @sparse_matrix_sum(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<f64>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<f64> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<8xf64>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 1 : index} : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.pointers %[[VAL_2]] {dimension = 1 : index} : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_2]] {dimension = 1 : index} : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_2]] : tensor<64x32xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_14:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f64>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_14]][] : memref<f64>
// CHECK:           %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_6]] to %[[VAL_5]] step %[[VAL_7]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (f64) {
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_17]], %[[VAL_7]] : index
// CHECK:             %[[VAL_21:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_22:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:             %[[VAL_23:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:             %[[VAL_24:.*]]:3 = scf.while (%[[VAL_25:.*]] = %[[VAL_19]], %[[VAL_26:.*]] = %[[VAL_22]], %[[VAL_27:.*]] = %[[VAL_18]]) : (index, index, f64) -> (index, index, f64) {
// CHECK:               %[[VAL_28:.*]] = arith.cmpi ult, %[[VAL_25]], %[[VAL_21]] : index
// CHECK:               %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_26]], %[[VAL_23]] : index
// CHECK:               %[[VAL_30:.*]] = arith.andi %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:               scf.condition(%[[VAL_30]]) %[[VAL_25]], %[[VAL_26]], %[[VAL_27]] : index, index, f64
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_31:.*]]: index, %[[VAL_32:.*]]: index, %[[VAL_33:.*]]: f64):
// CHECK:               %[[VAL_34:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_31]]] : memref<?xindex>
// CHECK:               %[[VAL_35:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_32]]] : memref<?xindex>
// CHECK:               %[[VAL_36:.*]] = arith.cmpi ult, %[[VAL_35]], %[[VAL_34]] : index
// CHECK:               %[[VAL_37:.*]] = arith.select %[[VAL_36]], %[[VAL_35]], %[[VAL_34]] : index
// CHECK:               %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_34]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = arith.cmpi eq, %[[VAL_35]], %[[VAL_37]] : index
// CHECK:               %[[VAL_40:.*]] = arith.andi %[[VAL_38]], %[[VAL_39]] : i1
// CHECK:               %[[VAL_41:.*]] = scf.if %[[VAL_40]] -> (f64) {
// CHECK:                 %[[VAL_42:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_31]]] : memref<?xf64>
// CHECK:                 %[[VAL_43:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_32]]] : memref<?xf64>
// CHECK:                 %[[VAL_44:.*]] = arith.addf %[[VAL_42]], %[[VAL_43]] : f64
// CHECK:                 %[[VAL_45:.*]] = arith.addf %[[VAL_33]], %[[VAL_44]] : f64
// CHECK:                 scf.yield %[[VAL_45]] : f64
// CHECK:               } else {
// CHECK:                 %[[VAL_46:.*]] = scf.if %[[VAL_38]] -> (f64) {
// CHECK:                   %[[VAL_47:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_31]]] : memref<?xf64>
// CHECK:                   %[[VAL_48:.*]] = arith.addf %[[VAL_33]], %[[VAL_47]] : f64
// CHECK:                   scf.yield %[[VAL_48]] : f64
// CHECK:                 } else {
// CHECK:                   %[[VAL_49:.*]] = scf.if %[[VAL_39]] -> (f64) {
// CHECK:                     %[[VAL_50:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_32]]] : memref<?xf64>
// CHECK:                     %[[VAL_51:.*]] = arith.addf %[[VAL_33]], %[[VAL_50]] : f64
// CHECK:                     scf.yield %[[VAL_51]] : f64
// CHECK:                   } else {
// CHECK:                     scf.yield %[[VAL_33]] : f64
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_52:.*]] : f64
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_53:.*]] : f64
// CHECK:               }
// CHECK:               %[[VAL_54:.*]] = arith.addi %[[VAL_31]], %[[VAL_7]] : index
// CHECK:               %[[VAL_55:.*]] = arith.select %[[VAL_38]], %[[VAL_54]], %[[VAL_31]] : index
// CHECK:               %[[VAL_56:.*]] = arith.addi %[[VAL_32]], %[[VAL_7]] : index
// CHECK:               %[[VAL_57:.*]] = arith.select %[[VAL_39]], %[[VAL_56]], %[[VAL_32]] : index
// CHECK:               scf.yield %[[VAL_55]], %[[VAL_57]], %[[VAL_58:.*]] : index, index, f64
// CHECK:             } attributes {"Emitted from" = "linalg.generic"}
// CHECK:             %[[VAL_59:.*]] = vector.insertelement %[[VAL_60:.*]]#2, %[[VAL_4]]{{\[}}%[[VAL_6]] : index] : vector<8xf64>
// CHECK:             %[[VAL_61:.*]] = scf.for %[[VAL_62:.*]] = %[[VAL_60]]#0 to %[[VAL_21]] step %[[VAL_3]] iter_args(%[[VAL_63:.*]] = %[[VAL_59]]) -> (vector<8xf64>) {
// CHECK:               %[[VAL_64:.*]] = affine.min #map(%[[VAL_21]], %[[VAL_62]]){{\[}}%[[VAL_3]]]
// CHECK:               %[[VAL_65:.*]] = vector.create_mask %[[VAL_64]] : vector<8xi1>
// CHECK:               %[[VAL_66:.*]] = vector.maskedload %[[VAL_10]]{{\[}}%[[VAL_62]]], %[[VAL_65]], %[[VAL_4]] : memref<?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
// CHECK:               %[[VAL_67:.*]] = arith.addf %[[VAL_63]], %[[VAL_66]] : vector<8xf64>
// CHECK:               %[[VAL_68:.*]] = arith.select %[[VAL_65]], %[[VAL_67]], %[[VAL_63]] : vector<8xi1>, vector<8xf64>
// CHECK:               scf.yield %[[VAL_68]] : vector<8xf64>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             %[[VAL_69:.*]] = scf.for %[[VAL_70:.*]] = %[[VAL_60]]#1 to %[[VAL_23]] step %[[VAL_3]] iter_args(%[[VAL_71:.*]] = %[[VAL_61]]) -> (vector<8xf64>) {
// CHECK:               %[[VAL_73:.*]] = affine.min #map(%[[VAL_23]], %[[VAL_70]]){{\[}}%[[VAL_3]]]
// CHECK:               %[[VAL_74:.*]] = vector.create_mask %[[VAL_73]] : vector<8xi1>
// CHECK:               %[[VAL_75:.*]] = vector.maskedload %[[VAL_13]]{{\[}}%[[VAL_70]]], %[[VAL_74]], %[[VAL_4]] : memref<?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
// CHECK:               %[[VAL_76:.*]] = arith.addf %[[VAL_71]], %[[VAL_75]] : vector<8xf64>
// CHECK:               %[[VAL_77:.*]] = arith.select %[[VAL_74]], %[[VAL_76]], %[[VAL_71]] : vector<8xi1>, vector<8xf64>
// CHECK:               scf.yield %[[VAL_77]] : vector<8xf64>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:             %[[VAL_78:.*]] = vector.reduction <add>, %[[VAL_69]] : vector<8xf64> into f64
// CHECK:             scf.yield %[[VAL_78]] : f64
// CHECK:           } {"Emitted from" = "linalg.generic"}
// CHECK:           memref.store %[[VAL_80:.*]], %[[VAL_14]][] : memref<f64>
// CHECK:           %[[VAL_81:.*]] = bufferization.to_tensor %[[VAL_14]] : memref<f64>
// CHECK:           return %[[VAL_81]] : tensor<f64>
// CHECK:         }
func.func @sparse_matrix_sum(%argx: tensor<f64>,
                             %arga: tensor<64x32xf64, #SparseMatrix>,
                             %argb: tensor<64x32xf64, #SparseMatrix>) -> tensor<f64> {
  %0 = linalg.generic #trait
     ins(%arga, %argb: tensor<64x32xf64, #SparseMatrix>,
                       tensor<64x32xf64, #SparseMatrix>)
      outs(%argx: tensor<f64>) {
      ^bb(%a: f64, %b: f64, %x: f64):
        %m = arith.addf %a, %b : f64
        %t = arith.addf %x, %m : f64
        linalg.yield %t : f64
  } -> tensor<f64>
  return %0 : tensor<f64>
}
