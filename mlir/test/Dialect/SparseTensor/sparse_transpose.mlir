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
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) -> tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = bufferization.alloc_tensor() : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           %[[VAL_4:.*]] = sparse_tensor.convert %[[VAL_0]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>>
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_4]] {dimension = 0 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> to memref<?xindex>
// CHECK:           %[[VAL_6:.*]] = sparse_tensor.indices %[[VAL_4]] {dimension = 0 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> to memref<?xindex>
// CHECK:           %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_4]] {dimension = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> to memref<?xindex>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.indices %[[VAL_4]] {dimension = 1 : index} : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_4]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>> to memref<?xf64>
// CHECK:           %[[VAL_10:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_1]]] : memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK:           scf.for %[[VAL_12:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] {
// CHECK:             %[[VAL_13:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:             %[[VAL_14:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_2]] : index
// CHECK:             %[[VAL_16:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK:             scf.for %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_2]] {
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK:               %[[VAL_19:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK:               sparse_tensor.insert %[[VAL_19]] into %[[VAL_3]]{{\[}}%[[VAL_13]], %[[VAL_18]]] : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = sparse_tensor.load %[[VAL_3]] hasInserts : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
// CHECK:           bufferization.dealloc_tensor %[[VAL_4]] : tensor<3x4xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], dimOrdering = affine_map<(d0, d1) -> (d1, d0)> }>>
// CHECK:           return %[[VAL_20]] : tensor<4x3xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
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
