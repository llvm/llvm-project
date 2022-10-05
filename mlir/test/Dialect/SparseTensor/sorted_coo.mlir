// RUN: mlir-opt %s -sparsification | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
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

// CHECK-LABEL: func.func @sparse_scale(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>) -> tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> {
// CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:     %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xf32>
// CHECK:         %[[VAL_6:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:         %[[VAL_7:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_7]] step %[[VAL_2]] {
// CHECK:           %[[VAL_9:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xf32>
// CHECK:           %[[VAL_10:.*]] = arith.mulf %[[VAL_9]], %[[VAL_3]] : f32
// CHECK:           memref.store %[[VAL_10]], %[[VAL_5]]{{\[}}%[[VAL_8]]] : memref<?xf32>
// CHECK:         }
// CHECK:         %[[VAL_11:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>
// CHECK:         return %[[VAL_11]] : tensor<?x?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>
// CHECK:       }
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

// CHECK-LABEL: func.func @matvec(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:    %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-SAME:    %[[VAL_2:.*]]: tensor<32xf64>) -> tensor<32xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK:         %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_1]] : memref<64xf64>
// CHECK:         %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32xf64>
// CHECK:         %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK:         %[[VAL_12:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         scf.for %[[VAL_13:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_4]] {
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:           %[[VAL_15:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_14]]] : memref<32xf64>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<?xf64>
// CHECK:           %[[VAL_18:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_16]]] : memref<64xf64>
// CHECK:           %[[VAL_19:.*]] = arith.mulf %[[VAL_17]], %[[VAL_18]] : f64
// CHECK:           %[[VAL_20:.*]] = arith.addf %[[VAL_15]], %[[VAL_19]] : f64
// CHECK:           memref.store %[[VAL_20]], %[[VAL_10]]{{\[}}%[[VAL_14]]] : memref<32xf64>
// CHECK:         }
// CHECK:         %[[VAL_21:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<32xf64>
// CHECK:         return %[[VAL_21]] : tensor<32xf64>
// CHECK:       }
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

// CHECK-LABEL: func.func @mateltmul(
// CHECK-SAME:    %[[VAL_0:.*0]]: tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:    %[[VAL_1:.*1]]: tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>>,
// CHECK-SAME:    %[[VAL_2:.*2]]: tensor<32x64xf64>) -> tensor<32x64xf64> {
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_8:.*]] = sparse_tensor.indices %[[VAL_0]] {dimension = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK-DAG:     %[[VAL_10:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_11:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 0 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_12:.*]] = sparse_tensor.indices %[[VAL_1]] {dimension = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xindex>
// CHECK-DAG:     %[[VAL_13:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK:         %[[VAL_14:.*]] = bufferization.to_memref %[[VAL_2]] : memref<32x64xf64>
// CHECK:         linalg.fill ins(%[[VAL_3]] : f64) outs(%[[VAL_14]] : memref<32x64xf64>)
// CHECK:         %[[VAL_15:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         %[[VAL_16:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:         %[[VAL_17:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK:         %[[VAL_18:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK:         %[[VAL_19:.*]]:2 = scf.while (%[[VAL_20:.*]] = %[[VAL_15]], %[[VAL_21:.*]] = %[[VAL_17]]) : (index, index) -> (index, index) {
// CHECK:           %[[VAL_22:.*]] = arith.cmpi ult, %[[VAL_20]], %[[VAL_16]] : index
// CHECK:           %[[VAL_23:.*]] = arith.cmpi ult, %[[VAL_21]], %[[VAL_18]] : index
// CHECK:           %[[VAL_24:.*]] = arith.andi %[[VAL_22]], %[[VAL_23]] : i1
// CHECK:           scf.condition(%[[VAL_24]]) %[[VAL_20]], %[[VAL_21]] : index, index
// CHECK:         } do {
// CHECK:         ^bb0(%[[VAL_25:.*]]: index, %[[VAL_26:.*]]: index):
// CHECK:           %[[VAL_27:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_25]]] : memref<?xindex>
// CHECK:           %[[VAL_28:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_26]]] : memref<?xindex>
// CHECK:           %[[VAL_29:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_27]] : index
// CHECK:           %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_28]], %[[VAL_27]] : index
// CHECK:           %[[VAL_31:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_30]] : index
// CHECK:           %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:           %[[VAL_33:.*]] = arith.andi %[[VAL_31]], %[[VAL_32]] : i1
// CHECK:           scf.if %[[VAL_33]] {
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_25]], %[[VAL_5]] : index
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_26]], %[[VAL_5]] : index
// CHECK:             %[[VAL_36:.*]]:2 = scf.while (%[[VAL_37:.*]] = %[[VAL_25]], %[[VAL_38:.*]] = %[[VAL_26]]) : (index, index) -> (index, index) {
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_37]], %[[VAL_34]] : index
// CHECK:               %[[VAL_40:.*]] = arith.cmpi ult, %[[VAL_38]], %[[VAL_35]] : index
// CHECK:               %[[VAL_41:.*]] = arith.andi %[[VAL_39]], %[[VAL_40]] : i1
// CHECK:               scf.condition(%[[VAL_41]]) %[[VAL_37]], %[[VAL_38]] : index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_42:.*]]: index, %[[VAL_43:.*]]: index):
// CHECK:               %[[VAL_44:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_42]]] : memref<?xindex>
// CHECK:               %[[VAL_45:.*]] = memref.load %[[VAL_12]]{{\[}}%[[VAL_43]]] : memref<?xindex>
// CHECK:               %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_45]], %[[VAL_44]] : index
// CHECK:               %[[VAL_47:.*]] = arith.select %[[VAL_46]], %[[VAL_45]], %[[VAL_44]] : index
// CHECK:               %[[VAL_48:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_47]] : index
// CHECK:               %[[VAL_49:.*]] = arith.cmpi eq, %[[VAL_45]], %[[VAL_47]] : index
// CHECK:               %[[VAL_50:.*]] = arith.andi %[[VAL_48]], %[[VAL_49]] : i1
// CHECK:               scf.if %[[VAL_50]] {
// CHECK:                 %[[VAL_51:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_42]]] : memref<?xf64>
// CHECK:                 %[[VAL_52:.*]] = memref.load %[[VAL_13]]{{\[}}%[[VAL_43]]] : memref<?xf64>
// CHECK:                 %[[VAL_53:.*]] = arith.mulf %[[VAL_51]], %[[VAL_52]] : f64
// CHECK:                 memref.store %[[VAL_53]], %[[VAL_14]]{{\[}}%[[VAL_30]], %[[VAL_47]]] : memref<32x64xf64>
// CHECK:               } else {
// CHECK:               }
// CHECK:               %[[VAL_54:.*]] = arith.cmpi eq, %[[VAL_44]], %[[VAL_47]] : index
// CHECK:               %[[VAL_55:.*]] = arith.addi %[[VAL_42]], %[[VAL_5]] : index
// CHECK:               %[[VAL_56:.*]] = arith.select %[[VAL_54]], %[[VAL_55]], %[[VAL_42]] : index
// CHECK:               %[[VAL_57:.*]] = arith.cmpi eq, %[[VAL_45]], %[[VAL_47]] : index
// CHECK:               %[[VAL_58:.*]] = arith.addi %[[VAL_43]], %[[VAL_5]] : index
// CHECK:               %[[VAL_59:.*]] = arith.select %[[VAL_57]], %[[VAL_58]], %[[VAL_43]] : index
// CHECK:               scf.yield %[[VAL_56]], %[[VAL_59]] : index, index
// CHECK:             }
// CHECK:           } else {
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = arith.cmpi eq, %[[VAL_27]], %[[VAL_30]] : index
// CHECK:           %[[VAL_61:.*]] = arith.addi %[[VAL_25]], %[[VAL_5]] : index
// CHECK:           %[[VAL_62:.*]] = arith.select %[[VAL_60]], %[[VAL_61]], %[[VAL_25]] : index
// CHECK:           %[[VAL_63:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:           %[[VAL_64:.*]] = arith.addi %[[VAL_26]], %[[VAL_5]] : index
// CHECK:           %[[VAL_65:.*]] = arith.select %[[VAL_63]], %[[VAL_64]], %[[VAL_26]] : index
// CHECK:           scf.yield %[[VAL_62]], %[[VAL_65]] : index, index
// CHECK:         }
// CHECK:         %[[VAL_66:.*]] = bufferization.to_tensor %[[VAL_14]] : memref<32x64xf64>
// CHECK:         return %[[VAL_66]] : tensor<32x64xf64>
// CHECK:       }
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
