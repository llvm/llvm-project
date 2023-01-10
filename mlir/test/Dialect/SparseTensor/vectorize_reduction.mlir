// RUN: mlir-opt %s -sparsification -cse -sparse-vectorization="vl=8" -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-ON
// RUN: mlir-opt %s -sparsification -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-OFF

// -----

// Check that we recognize a reduction with a mul operator.
// We use two dimensions here to check that the vectorization
// is not affected by how the outer loop is layed out.
// In other words, we should be able to vectorize the sparse inner loop
// regardless of whether the outer loop is dense or sparse.
//
// For this particular test, we expect:
// With vectorization on:
// dense scf.for
//   init vector_accumulator = {scalar_accumulator, 1.0, 1.0, ...}
//   sparse scf.for
//     vectorized mul in vector_accumulator, vector_input
//   horizontal reduction of the vector_accumulator to scalar_accumulator
// final store of scalar_accumulaor
//
// With vectorization off:
// dense scf.for
//   sparse scf.for
//     mul in accumulator
// final store
//
// CHECK-ON-LABEL:   func.func @sparse_product_reduction_dense_sparse(
// CHECK-ON-SAME:                                                     %[[VAL_0:.*]]: tensor<f64>,
// CHECK-ON-SAME:                                                     %[[VAL_1:.*]]: tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<f64> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<1.000000e+00> : vector<8xf64>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<8xf64>
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-ON-DAG:       %[[VAL_7:.*]] = tensor.dim %[[VAL_1]], %[[VAL_5]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-ON:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK-ON:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f64>
// CHECK-ON:           %[[VAL_11:.*]] = memref.load %[[VAL_10]][] : memref<f64>
// CHECK-ON:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (f64) {
// CHECK-ON:             %[[VAL_15:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-ON:             %[[VAL_16:.*]] = arith.addi %[[VAL_13]], %[[VAL_6]] : index
// CHECK-ON:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_16]]] : memref<?xindex>
// CHECK-ON:             %[[VAL_18:.*]] = vector.insertelement %[[VAL_14]], %[[VAL_3]]{{\[}}%[[VAL_5]] : index] : vector<8xf64>
// CHECK-ON:             %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_2]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (vector<8xf64>) {
// CHECK-ON:               %[[VAL_22:.*]] = affine.min #map(%[[VAL_17]], %[[VAL_20]]){{\[}}%[[VAL_2]]]
// CHECK-ON:               %[[VAL_23:.*]] = vector.create_mask %[[VAL_22]] : vector<8xi1>
// CHECK-ON:               %[[VAL_24:.*]] = vector.maskedload %[[VAL_9]]{{\[}}%[[VAL_20]]], %[[VAL_23]], %[[VAL_4]] : memref<?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
// CHECK-ON:               %[[VAL_25:.*]] = arith.mulf %[[VAL_21]], %[[VAL_24]] : vector<8xf64>
// CHECK-ON:               %[[VAL_26:.*]] = arith.select %[[VAL_23]], %[[VAL_25]], %[[VAL_21]] : vector<8xi1>, vector<8xf64>
// CHECK-ON:               scf.yield %[[VAL_26]] : vector<8xf64>
// CHECK-ON:             } {"Emitted from" = "linalg.generic"}
// CHECK-ON:             %[[VAL_27:.*]] = vector.reduction <mul>, %[[VAL_28:.*]] : vector<8xf64> into f64
// CHECK-ON:             scf.yield %[[VAL_27]] : f64
// CHECK-ON:           } {"Emitted from" = "linalg.generic"}
// CHECK-ON:           memref.store %[[VAL_29:.*]], %[[VAL_10]][] : memref<f64>
// CHECK-ON:           %[[VAL_30:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<f64>
// CHECK-ON:           return %[[VAL_30]] : tensor<f64>
// CHECK-ON:         }
//
// CHECK-OFF-LABEL:   func.func @sparse_product_reduction_dense_sparse(
// CHECK-OFF-SAME:                                                     %[[VAL_0:.*]]: tensor<f64>,
// CHECK-OFF-SAME:                                                     %[[VAL_1:.*]]: tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<f64> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = tensor.dim %[[VAL_1]], %[[VAL_2]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK-OFF:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f64>
// CHECK-OFF:           %[[VAL_8:.*]] = memref.load %[[VAL_7]][] : memref<f64>
// CHECK-OFF:           %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_2]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (f64) {
// CHECK-OFF:             %[[VAL_12:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_10]]] : memref<?xindex>
// CHECK-OFF:             %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_3]] : index
// CHECK-OFF:             %[[VAL_14:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-OFF:             %[[VAL_15:.*]] = scf.for %[[VAL_16:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_3]] iter_args(%[[VAL_17:.*]] = %[[VAL_11]]) -> (f64) {
// CHECK-OFF:               %[[VAL_18:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_16]]] : memref<?xf64>
// CHECK-OFF:               %[[VAL_19:.*]] = arith.mulf %[[VAL_17]], %[[VAL_18]] : f64
// CHECK-OFF:               scf.yield %[[VAL_19]] : f64
// CHECK-OFF:             } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:             scf.yield %[[VAL_20:.*]] : f64
// CHECK-OFF:           } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:           memref.store %[[VAL_21:.*]], %[[VAL_7]][] : memref<f64>
// CHECK-OFF:           %[[VAL_22:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<f64>
// CHECK-OFF:           return %[[VAL_22]] : tensor<f64>
// CHECK-OFF:         }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["dense","compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a (in)
    affine_map<(i,j) -> ()>      // x (out)
  ],
  iterator_types = ["reduction", "reduction"]
}

func.func @sparse_product_reduction_dense_sparse(%argx: tensor<f64>,
                             %arga: tensor<?x128xf64, #SparseVector>)
 -> tensor<f64> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?x128xf64, #SparseVector>)
      outs(%argx: tensor<f64>) {
      ^bb(%a: f64, %x: f64):
        %t = arith.mulf %x, %a: f64
        linalg.yield %t : f64
  } -> tensor<f64>
  return %0 : tensor<f64>
}

// -----

// Same as sparse_product_reduction_dense_sparse but with the outer loop being sparse.
//
// CHECK-ON-LABEL:   func.func @sparse_product_reduction_sparse_sparse(
// CHECK-ON-SAME:                                                      %[[VAL_0:.*]]: tensor<f64>,
// CHECK-ON-SAME:                                                      %[[VAL_1:.*]]: tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) -> tensor<f64> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<1.000000e+00> : vector<8xf64>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant dense<0.000000e+00> : vector<8xf64>
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_8:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_9:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf64>
// CHECK-ON:           %[[VAL_10:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f64>
// CHECK-ON:           %[[VAL_11:.*]] = memref.load %[[VAL_10]][] : memref<f64>
// CHECK-ON:           %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_13:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_12]] to %[[VAL_13]] step %[[VAL_6]] iter_args(%[[VAL_16:.*]] = %[[VAL_11]]) -> (f64) {
// CHECK-ON:             %[[VAL_17:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-ON:             %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_6]] : index
// CHECK-ON:             %[[VAL_19:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_18]]] : memref<?xindex>
// CHECK-ON:             %[[VAL_20:.*]] = vector.insertelement %[[VAL_16]], %[[VAL_3]]{{\[}}%[[VAL_5]] : index] : vector<8xf64>
// CHECK-ON:             %[[VAL_21:.*]] = scf.for %[[VAL_22:.*]] = %[[VAL_17]] to %[[VAL_19]] step %[[VAL_2]] iter_args(%[[VAL_23:.*]] = %[[VAL_20]]) -> (vector<8xf64>) {
// CHECK-ON:               %[[VAL_24:.*]] = affine.min #map(%[[VAL_19]], %[[VAL_22]]){{\[}}%[[VAL_2]]]
// CHECK-ON:               %[[VAL_25:.*]] = vector.create_mask %[[VAL_24]] : vector<8xi1>
// CHECK-ON:               %[[VAL_26:.*]] = vector.maskedload %[[VAL_9]]{{\[}}%[[VAL_22]]], %[[VAL_25]], %[[VAL_4]] : memref<?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
// CHECK-ON:               %[[VAL_27:.*]] = arith.mulf %[[VAL_23]], %[[VAL_26]] : vector<8xf64>
// CHECK-ON:               %[[VAL_28:.*]] = arith.select %[[VAL_25]], %[[VAL_27]], %[[VAL_23]] : vector<8xi1>, vector<8xf64>
// CHECK-ON:               scf.yield %[[VAL_28]] : vector<8xf64>
// CHECK-ON:             } {"Emitted from" = "linalg.generic"}
// CHECK-ON:             %[[VAL_29:.*]] = vector.reduction <mul>, %[[VAL_30:.*]] : vector<8xf64> into f64
// CHECK-ON:             scf.yield %[[VAL_29]] : f64
// CHECK-ON:           } {"Emitted from" = "linalg.generic"}
// CHECK-ON:           memref.store %[[VAL_31:.*]], %[[VAL_10]][] : memref<f64>
// CHECK-ON:           %[[VAL_32:.*]] = bufferization.to_tensor %[[VAL_10]] : memref<f64>
// CHECK-ON:           return %[[VAL_32]] : tensor<f64>
// CHECK-ON:         }
//
// CHECK-OFF-LABEL:   func.func @sparse_product_reduction_sparse_sparse(
// CHECK-OFF-SAME:                                                     %[[VAL_0:.*]]: tensor<f64>,
// CHECK-OFF-SAME:                                                     %[[VAL_1:.*]]: tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) -> tensor<f64> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 1 : index} : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_6:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?x128xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>> to memref<?xf64>
// CHECK-OFF:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f64>
// CHECK-OFF:           %[[VAL_8:.*]] = memref.load %[[VAL_7]][] : memref<f64>
// CHECK-OFF:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_10:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_3]] iter_args(%[[VAL_13:.*]] = %[[VAL_8]]) -> (f64) {
// CHECK-OFF:             %[[VAL_14:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_12]]] : memref<?xindex>
// CHECK-OFF:             %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_3]] : index
// CHECK-OFF:             %[[VAL_16:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-OFF:             %[[VAL_17:.*]] = scf.for %[[VAL_18:.*]] = %[[VAL_14]] to %[[VAL_16]] step %[[VAL_3]] iter_args(%[[VAL_19:.*]] = %[[VAL_13]]) -> (f64) {
// CHECK-OFF:               %[[VAL_20:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_18]]] : memref<?xf64>
// CHECK-OFF:               %[[VAL_21:.*]] = arith.mulf %[[VAL_19]], %[[VAL_20]] : f64
// CHECK-OFF:               scf.yield %[[VAL_21]] : f64
// CHECK-OFF:             } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:             scf.yield %[[VAL_22:.*]] : f64
// CHECK-OFF:           } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:           memref.store %[[VAL_23:.*]], %[[VAL_7]][] : memref<f64>
// CHECK-OFF:           %[[VAL_24:.*]] = bufferization.to_tensor %[[VAL_7]] : memref<f64>
// CHECK-OFF:           return %[[VAL_24]] : tensor<f64>
// CHECK-OFF:         }
#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed","compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a (in)
    affine_map<(i,j) -> ()>      // x (out)
  ],
  iterator_types = ["reduction", "reduction"]
}

func.func @sparse_product_reduction_sparse_sparse(%argx: tensor<f64>,
                             %arga: tensor<?x128xf64, #SparseVector>)
 -> tensor<f64> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?x128xf64, #SparseVector>)
      outs(%argx: tensor<f64>) {
      ^bb(%a: f64, %x: f64):
        %t = arith.mulf %x, %a: f64
        linalg.yield %t : f64
  } -> tensor<f64>
  return %0 : tensor<f64>
}

// -----

// sparse_product_reduction_dense_sparse and
// sparse_product_reduction_sparse_sparse established that the outer loop
// doesn't matter for vectorization.
// As a result from this point forward, use tensors with fewer dimensions.

// Check that we vectorize reductions with ori.
// Note: The weird element type here is to check that we create the right
// constant type for the pass-through value.
// CHECK-ON-LABEL:   func.func @sparse_reduction_ori(
// CHECK-ON-SAME:                                    %[[VAL_0:.*]]: tensor<i13>,
// CHECK-ON-SAME:                                    %[[VAL_1:.*]]: tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i13> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi13>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi13>
// CHECK-ON:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i13>
// CHECK-ON:           %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i13>
// CHECK-ON:           %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_12:.*]] = vector.broadcast %[[VAL_9]] : i13 to vector<8xi13>
// CHECK-ON:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi13>) {
// CHECK-ON:             %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:             %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:             %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xi13>, vector<8xi1>, vector<8xi13> into vector<8xi13>
// CHECK-ON:             %[[VAL_19:.*]] = arith.ori %[[VAL_15]], %[[VAL_18]] : vector<8xi13>
// CHECK-ON:             %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi13>
// CHECK-ON:             scf.yield %[[VAL_20]] : vector<8xi13>
// CHECK-ON:           } {"Emitted from" = "linalg.generic"}
// CHECK-ON:           %[[VAL_21:.*]] = vector.reduction <or>, %[[VAL_22:.*]] : vector<8xi13> into i13
// CHECK-ON:           memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i13>
// CHECK-ON:           %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i13>
// CHECK-ON:           return %[[VAL_23]] : tensor<i13>
// CHECK-ON:         }
//
// CHECK-OFF-LABEL:   func.func @sparse_reduction_ori(
// CHECK-OFF-SAME:                                    %[[VAL_0:.*]]: tensor<i13>,
// CHECK-OFF-SAME:                                    %[[VAL_1:.*]]: tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i13> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi13>
// CHECK-OFF:           %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i13>
// CHECK-OFF:           %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i13>
// CHECK-OFF:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i13) {
// CHECK-OFF:             %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi13>
// CHECK-OFF:             %[[VAL_14:.*]] = arith.ori %[[VAL_12]], %[[VAL_13]] : i13
// CHECK-OFF:             scf.yield %[[VAL_14]] : i13
// CHECK-OFF:           } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:           memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i13>
// CHECK-OFF:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i13>
// CHECK-OFF:           return %[[VAL_16]] : tensor<i13>
// CHECK-OFF:         }
#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_ori(%argx: tensor<i13>,
                             %arga: tensor<?xi13, #SparseVector>)
 -> tensor<i13> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi13, #SparseVector>)
      outs(%argx: tensor<i13>) {
      ^bb(%a: i13, %x: i13):
        %t = arith.ori %x, %a: i13
        linalg.yield %t : i13
  } -> tensor<i13>
  return %0 : tensor<i13>
}

// -----

// Same test as sparse_reduction_ori except that the accumulator is on the
// rhs of the operation.
// This checks that we can recognize a reduction irrespective to where the
// accumalator appears on commutative operations.

// CHECK-ON-LABEL:   func.func @sparse_reduction_ori_accumulator_on_rhs(
// CHECK-ON-SAME:                                    %[[VAL_0:.*]]: tensor<i13>,
// CHECK-ON-SAME:                                    %[[VAL_1:.*]]: tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i13> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi13>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi13>
// CHECK-ON:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i13>
// CHECK-ON:           %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i13>
// CHECK-ON:           %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_12:.*]] = vector.broadcast %[[VAL_9]] : i13 to vector<8xi13>
// CHECK-ON:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi13>) {
// CHECK-ON:             %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:             %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:             %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xi13>, vector<8xi1>, vector<8xi13> into vector<8xi13>
// CHECK-ON:             %[[VAL_19:.*]] = arith.ori %[[VAL_18]], %[[VAL_15]] : vector<8xi13>
// CHECK-ON:             %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi13>
// CHECK-ON:             scf.yield %[[VAL_20]] : vector<8xi13>
// CHECK-ON:           } {"Emitted from" = "linalg.generic"}
// CHECK-ON:           %[[VAL_21:.*]] = vector.reduction <or>, %[[VAL_22:.*]] : vector<8xi13> into i13
// CHECK-ON:           memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i13>
// CHECK-ON:           %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i13>
// CHECK-ON:           return %[[VAL_23]] : tensor<i13>
// CHECK-ON:         }
//
// CHECK-OFF-LABEL:   func.func @sparse_reduction_ori_accumulator_on_rhs(
// CHECK-OFF-SAME:                                    %[[VAL_0:.*]]: tensor<i13>,
// CHECK-OFF-SAME:                                    %[[VAL_1:.*]]: tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i13> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi13>
// CHECK-OFF:           %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i13>
// CHECK-OFF:           %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i13>
// CHECK-OFF:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i13) {
// CHECK-OFF:             %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi13>
// CHECK-OFF:             %[[VAL_14:.*]] = arith.ori %[[VAL_13]], %[[VAL_12]] : i13
// CHECK-OFF:             scf.yield %[[VAL_14]] : i13
// CHECK-OFF:           } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:           memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i13>
// CHECK-OFF:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i13>
// CHECK-OFF:           return %[[VAL_16]] : tensor<i13>
// CHECK-OFF:         }
#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_ori_accumulator_on_rhs(%argx: tensor<i13>,
                             %arga: tensor<?xi13, #SparseVector>)
 -> tensor<i13> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi13, #SparseVector>)
      outs(%argx: tensor<i13>) {
      ^bb(%a: i13, %x: i13):
        %t = arith.ori %a, %x: i13
        linalg.yield %t : i13
  } -> tensor<i13>
  return %0 : tensor<i13>
}

// -----

// Check that we vectorize reduction with subi.
//
// CHECK-ON-LABEL:   func.func @sparse_reduction_subi(
// CHECK-ON-SAME:                                     %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:                                     %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-ON:           %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-ON:           %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK-ON:           %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:           %[[VAL_12:.*]] = vector.insertelement %[[VAL_9]], %[[VAL_4]]{{\[}}%[[VAL_3]] : index] : vector<8xi32>
// CHECK-ON:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi32>) {
// CHECK-ON:             %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:             %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:             %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_4]] : memref<?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
// CHECK-ON:             %[[VAL_19:.*]] = arith.subi %[[VAL_15]], %[[VAL_18]] : vector<8xi32>
// CHECK-ON:             %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi32>
// CHECK-ON:             scf.yield %[[VAL_20]] : vector<8xi32>
// CHECK-ON:           } {"Emitted from" = "linalg.generic"}
// CHECK-ON:           %[[VAL_21:.*]] = vector.reduction <add>, %[[VAL_22:.*]] : vector<8xi32> into i32
// CHECK-ON:           memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i32>
// CHECK-ON:           %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i32>
// CHECK-ON:           return %[[VAL_23]] : tensor<i32>
// CHECK-ON:         }
//
// CHECK-OFF-LABEL:   func.func @sparse_reduction_subi(
// CHECK-OFF-SAME:                                     %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:                                     %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-OFF:           %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-OFF:           %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK-OFF:           %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:           %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i32) {
// CHECK-OFF:             %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK-OFF:             %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK-OFF:             scf.yield %[[VAL_14]] : i32
// CHECK-OFF:           } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:           memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i32>
// CHECK-OFF:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i32>
// CHECK-OFF:           return %[[VAL_16]] : tensor<i32>
// CHECK-OFF:         }
#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_subi(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %t = arith.subi %x, %a: i32
        linalg.yield %t : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// From this point forward, we essentially have the same test for all
// arithmetic operation. This is for a code coverage perspective.

// Check that we vectorize xor.
// CHECK-ON-LABEL: func.func @sparse_reduction_xor(
// CHECK-ON-SAME: %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME: %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-ON:  %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:  %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON:  %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:  %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:  %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:  %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-ON:  %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-ON:  %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK-ON:  %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:  %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:  %[[VAL_12:.*]] = vector.insertelement %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_4]] : index] : vector<8xi32>
// CHECK-ON:  %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi32>) {
// CHECK-ON:    %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:    %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:    %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
// CHECK-ON:    %[[VAL_19:.*]] = arith.xori %[[VAL_15]], %[[VAL_18]] : vector<8xi32>
// CHECK-ON:    %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi32>
// CHECK-ON:    scf.yield %[[VAL_20]] : vector<8xi32>
// CHECK-ON:  } {"Emitted from" = "linalg.generic"}
// CHECK-ON:  %[[VAL_21:.*]] = vector.reduction <xor>, %[[VAL_22:.*]] : vector<8xi32> into i32
// CHECK-ON:  memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i32>
// CHECK-ON:  %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i32>
// CHECK-ON:  return %[[VAL_23]] : tensor<i32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_xor(
// CHECK-OFF-SAME:  %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:  %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.xori %[[VAL_12]], %[[VAL_13]] : i32
// CHECK-OFF:     scf.yield %[[VAL_14]] : i32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<i32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_xor(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %t = arith.xori %x, %a: i32
        linalg.yield %t : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}

// -----
// Check that we vectorize and.
// CHECK-ON-LABEL: func.func @sparse_reduction_and(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-ON:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:   %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-ON:   %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-ON:   %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK-ON:   %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_12:.*]] = vector.broadcast %[[VAL_9]] : i32 to vector<8xi32>
// CHECK-ON:   %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi32>) {
// CHECK-ON:     %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:     %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:     %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
// CHECK-ON:     %[[VAL_19:.*]] = arith.andi %[[VAL_15]], %[[VAL_18]] : vector<8xi32>
// CHECK-ON:     %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi32>
// CHECK-ON:     scf.yield %[[VAL_20]] : vector<8xi32>
// CHECK-ON:   } {"Emitted from" = "linalg.generic"}
// CHECK-ON:   %[[VAL_21:.*]] = vector.reduction <and>, %[[VAL_22:.*]] : vector<8xi32> into i32
// CHECK-ON:   memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i32>
// CHECK-ON:   %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i32>
// CHECK-ON:   return %[[VAL_23]] : tensor<i32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_and(
// CHECK-OFF-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.andi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK-OFF:     scf.yield %[[VAL_14]] : i32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<i32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_and(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %t = arith.andi %x, %a: i32
        linalg.yield %t : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}

// -----
// Check that we vectorize muli.
// CHECK-ON-LABEL: func.func @sparse_reduction_muli(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-ON:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:   %[[VAL_3:.*]] = arith.constant dense<1> : vector<8xi32>
// CHECK-ON:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:   %[[VAL_5:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON:   %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:   %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-ON:   %[[VAL_9:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-ON:   %[[VAL_10:.*]] = memref.load %[[VAL_9]][] : memref<i32>
// CHECK-ON:   %[[VAL_11:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_12:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_6]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_13:.*]] = vector.insertelement %[[VAL_10]], %[[VAL_3]]{{\[}}%[[VAL_4]] : index] : vector<8xi32>
// CHECK-ON:   %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_11]] to %[[VAL_12]] step %[[VAL_2]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (vector<8xi32>) {
// CHECK-ON:     %[[VAL_17:.*]] = affine.min #map(%[[VAL_12]], %[[VAL_15]]){{\[}}%[[VAL_2]]]
// CHECK-ON:     %[[VAL_18:.*]] = vector.create_mask %[[VAL_17]] : vector<8xi1>
// CHECK-ON:     %[[VAL_19:.*]] = vector.maskedload %[[VAL_8]]{{\[}}%[[VAL_15]]], %[[VAL_18]], %[[VAL_5]] : memref<?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
// CHECK-ON:     %[[VAL_20:.*]] = arith.muli %[[VAL_16]], %[[VAL_19]] : vector<8xi32>
// CHECK-ON:     %[[VAL_21:.*]] = arith.select %[[VAL_18]], %[[VAL_20]], %[[VAL_16]] : vector<8xi1>, vector<8xi32>
// CHECK-ON:     scf.yield %[[VAL_21]] : vector<8xi32>
// CHECK-ON:   } {"Emitted from" = "linalg.generic"}
// CHECK-ON:   %[[VAL_22:.*]] = vector.reduction <mul>, %[[VAL_23:.*]] : vector<8xi32> into i32
// CHECK-ON:   memref.store %[[VAL_22]], %[[VAL_9]][] : memref<i32>
// CHECK-ON:   %[[VAL_24:.*]] = bufferization.to_tensor %[[VAL_9]] : memref<i32>
// CHECK-ON:   return %[[VAL_24]] : tensor<i32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_muli(
// CHECK-OFF-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.muli %[[VAL_12]], %[[VAL_13]] : i32
// CHECK-OFF:     scf.yield %[[VAL_14]] : i32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<i32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_muli(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %t = arith.muli %x, %a: i32
        linalg.yield %t : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}

// -----
// Check that we vectorize addi.
// CHECK-ON-LABEL: func.func @sparse_reduction_addi(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-ON:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:   %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-ON:   %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-ON:   %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<i32>
// CHECK-ON:   %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_12:.*]] = vector.insertelement %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_4]] : index] : vector<8xi32>
// CHECK-ON:   %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xi32>) {
// CHECK-ON:     %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:     %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:     %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xi32>, vector<8xi1>, vector<8xi32> into vector<8xi32>
// CHECK-ON:     %[[VAL_19:.*]] = arith.addi %[[VAL_15]], %[[VAL_18]] : vector<8xi32>
// CHECK-ON:     %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xi32>
// CHECK-ON:     scf.yield %[[VAL_20]] : vector<8xi32>
// CHECK-ON:   } {"Emitted from" = "linalg.generic"}
// CHECK-ON:   %[[VAL_21:.*]] = vector.reduction <add>, %[[VAL_22:.*]] : vector<8xi32> into i32
// CHECK-ON:   memref.store %[[VAL_21]], %[[VAL_8]][] : memref<i32>
// CHECK-ON:   %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<i32>
// CHECK-ON:   return %[[VAL_23]] : tensor<i32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_addi(
// CHECK-OFF-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<i32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xi32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<i32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (i32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xi32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK-OFF:     scf.yield %[[VAL_14]] : i32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<i32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<i32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<i32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_addi(%argx: tensor<i32>,
                             %arga: tensor<?xi32, #SparseVector>)
 -> tensor<i32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xi32, #SparseVector>)
      outs(%argx: tensor<i32>) {
      ^bb(%a: i32, %x: i32):
        %t = arith.addi %x, %a: i32
        linalg.yield %t : i32
  } -> tensor<i32>
  return %0 : tensor<i32>
}

// -----
// Check that we vectorize subf.
// CHECK-ON-LABEL: func.func @sparse_reduction_subf(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<f32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<f32> {
// CHECK-ON:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:   %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-ON:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xf32>
// CHECK-ON:   %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f32>
// CHECK-ON:   %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<f32>
// CHECK-ON:   %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_12:.*]] = vector.insertelement %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_4]] : index] : vector<8xf32>
// CHECK-ON:   %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xf32>) {
// CHECK-ON:     %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:     %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:     %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-ON:     %[[VAL_19:.*]] = arith.subf %[[VAL_15]], %[[VAL_18]] : vector<8xf32>
// CHECK-ON:     %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xf32>
// CHECK-ON:     scf.yield %[[VAL_20]] : vector<8xf32>
// CHECK-ON:   } {"Emitted from" = "linalg.generic"}
// CHECK-ON:   %[[VAL_21:.*]] = vector.reduction <add>, %[[VAL_22:.*]] : vector<8xf32> into f32
// CHECK-ON:   memref.store %[[VAL_21]], %[[VAL_8]][] : memref<f32>
// CHECK-ON:   %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<f32>
// CHECK-ON:   return %[[VAL_23]] : tensor<f32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_subf(
// CHECK-OFF-SAME:   %[[VAL_0:.*]]: tensor<f32>,
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<f32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xf32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<f32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (f32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.subf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK-OFF:     scf.yield %[[VAL_14]] : f32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<f32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<f32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<f32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_subf(%argx: tensor<f32>,
                             %arga: tensor<?xf32, #SparseVector>)
 -> tensor<f32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xf32, #SparseVector>)
      outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %x: f32):
        %t = arith.subf %x, %a: f32
        linalg.yield %t : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// Check that we vectorize addf.
// CHECK-ON-LABEL: func.func @sparse_reduction_addf(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<f32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<f32> {
// CHECK-ON:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON:   %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-ON:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xf32>
// CHECK-ON:   %[[VAL_8:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f32>
// CHECK-ON:   %[[VAL_9:.*]] = memref.load %[[VAL_8]][] : memref<f32>
// CHECK-ON:   %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_4]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_11:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_5]]] : memref<?xindex>
// CHECK-ON:   %[[VAL_12:.*]] = vector.insertelement %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_4]] : index] : vector<8xf32>
// CHECK-ON:   %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_2]] iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (vector<8xf32>) {
// CHECK-ON:     %[[VAL_16:.*]] = affine.min #map(%[[VAL_11]], %[[VAL_14]]){{\[}}%[[VAL_2]]]
// CHECK-ON:     %[[VAL_17:.*]] = vector.create_mask %[[VAL_16]] : vector<8xi1>
// CHECK-ON:     %[[VAL_18:.*]] = vector.maskedload %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_17]], %[[VAL_3]] : memref<?xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
// CHECK-ON:     %[[VAL_19:.*]] = arith.addf %[[VAL_15]], %[[VAL_18]] : vector<8xf32>
// CHECK-ON:     %[[VAL_20:.*]] = arith.select %[[VAL_17]], %[[VAL_19]], %[[VAL_15]] : vector<8xi1>, vector<8xf32>
// CHECK-ON:     scf.yield %[[VAL_20]] : vector<8xf32>
// CHECK-ON:   } {"Emitted from" = "linalg.generic"}
// CHECK-ON:   %[[VAL_21:.*]] = vector.reduction <add>, %[[VAL_22:.*]] : vector<8xf32> into f32
// CHECK-ON:   memref.store %[[VAL_21]], %[[VAL_8]][] : memref<f32>
// CHECK-ON:   %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_8]] : memref<f32>
// CHECK-ON:   return %[[VAL_23]] : tensor<f32>
// CHECK-ON: }
//
// CHECK-OFF-LABEL: func.func @sparse_reduction_addf(
// CHECK-OFF-SAME:    %[[VAL_0:.*]]: tensor<f32>,
// CHECK-OFF-SAME:    %[[VAL_1:.*]]: tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<f32> {
// CHECK-OFF:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.pointers %[[VAL_1]] {dimension = 0 : index} : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>> to memref<?xf32>
// CHECK-OFF:   %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_0]] : memref<f32>
// CHECK-OFF:   %[[VAL_7:.*]] = memref.load %[[VAL_6]][] : memref<f32>
// CHECK-OFF:   %[[VAL_8:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_2]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_9:.*]] = memref.load %[[VAL_4]]{{\[}}%[[VAL_3]]] : memref<?xindex>
// CHECK-OFF:   %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_3]] iter_args(%[[VAL_12:.*]] = %[[VAL_7]]) -> (f32) {
// CHECK-OFF:     %[[VAL_13:.*]] = memref.load %[[VAL_5]]{{\[}}%[[VAL_11]]] : memref<?xf32>
// CHECK-OFF:     %[[VAL_14:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : f32
// CHECK-OFF:     scf.yield %[[VAL_14]] : f32
// CHECK-OFF:   } {"Emitted from" = "linalg.generic"}
// CHECK-OFF:   memref.store %[[VAL_15:.*]], %[[VAL_6]][] : memref<f32>
// CHECK-OFF:   %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_6]] : memref<f32>
// CHECK-OFF:   return %[[VAL_16]] : tensor<f32>
// CHECK-OFF: }

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"]
}

func.func @sparse_reduction_addf(%argx: tensor<f32>,
                             %arga: tensor<?xf32, #SparseVector>)
 -> tensor<f32> {
  %0 = linalg.generic #trait
     ins(%arga: tensor<?xf32, #SparseVector>)
      outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %x: f32):
        %t = arith.addf %x, %a: f32
        linalg.yield %t : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}
