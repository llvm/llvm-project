// RUN: mlir-opt %s -sparsification | FileCheck %s

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#transpose_trait = {
  indexing_maps = [
    affine_map<(i,j) -> (j,i)>,  // A
    affine_map<(i,j) -> (i,j)>   // X
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(j,i)"
}

// TODO: improve auto-conversion followed by yield

// CHECK-LABEL:   func.func @sparse_transpose_auto(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>) -> tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = bufferization.alloc_tensor() : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.convert %[[VAL_0]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>> to tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_5]], %[[VAL_1]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_5]], %[[VAL_1]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_5]], %[[VAL_2]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.indices %[[VAL_5]], %[[VAL_2]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_5]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = memref.alloca(%[[VAL_3]]) : memref<?xindex>
// CHECK:           %[[VAL_12:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_15:.*]] = %[[VAL_13]] to %[[VAL_14]] step %[[VAL_2]] {
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             memref.store %[[VAL_16]], %[[VAL_11]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : index
// CHECK:             %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_20:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_2]] {
// CHECK:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_20]]] : memref<?xindex>
// CHECK:               memref.store %[[VAL_21]], %[[VAL_11]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_20]]] : memref<?xf64>
// CHECK:               memref.store %[[VAL_22]], %[[VAL_12]][] : memref<f64>
// CHECK:               sparse_tensor.lex_insert %[[VAL_4]], %[[VAL_11]], %[[VAL_12]] : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>, memref<?xindex>, memref<f64>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_23:.*]] = sparse_tensor.load %[[VAL_4]] hasInserts : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:           sparse_tensor.release %[[VAL_5]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)>, pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:           return %[[VAL_23]] : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:         }
func.func @sparse_transpose_auto(%arga: tensor<3x4xf64, #DCSR>)
                                     -> tensor<4x3xf64, #DCSR> {
  %i = bufferization.alloc_tensor() : tensor<4x3xf64, #DCSR>
  %0 = linalg.generic #transpose_trait
     ins(%arga: tensor<3x4xf64, #DCSR>)
     outs(%i: tensor<4x3xf64, #DCSR>) {
     ^bb(%a: f64, %x: f64):
       linalg.yield %a : f64
  } -> tensor<4x3xf64, #DCSR>
  return %0 : tensor<4x3xf64, #DCSR>
}
