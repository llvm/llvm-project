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

//
// Two kernels that operate on SortedCOO format.
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
