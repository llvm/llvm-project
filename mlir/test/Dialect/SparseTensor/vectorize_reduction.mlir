// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse -sparse-vectorization="vl=8" -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-ON
// RUN: mlir-opt %s --sparse-reinterpret-map -sparsification -cse -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-OFF

// -----

// Check that we vectorize reductions with ori.

// CHECK-ON-LABEL:   func.func @sparse_reduction_ori(
// CHECK-ON-SAME:      %[[VAL_0:.*]]: tensor<i13>,
// CHECK-ON-SAME:      %[[VAL_1:.*]]: tensor<?xi13, #sparse{{[0-9]*}}>) -> tensor<i13> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi13>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xi13>
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
// CHECK-OFF-SAME:      %[[VAL_0:.*]]: tensor<i13>,
// CHECK-OFF-SAME:      %[[VAL_1:.*]]: tensor<?xi13, #sparse{{[0-9]*}}>) -> tensor<i13> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xi13>
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
#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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
// rhs of the operation. This checks that we can recognize a reduction
// irrespective to where the accumulator appears on commutative operations.

// CHECK-ON-LABEL:   func.func @sparse_reduction_ori_accumulator_on_rhs(
// CHECK-ON-SAME:      %[[VAL_0:.*]]: tensor<i13>,
// CHECK-ON-SAME:      %[[VAL_1:.*]]: tensor<?xi13, #sparse{{[0-9]*}}>) -> tensor<i13> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi13>
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xi13>
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
// CHECK-OFF-SAME:      %[[VAL_0:.*]]: tensor<i13>,
// CHECK-OFF-SAME:      %[[VAL_1:.*]]: tensor<?xi13, #sparse{{[0-9]*}}>) -> tensor<i13> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi13, #sparse{{[0-9]*}}> to memref<?xi13>
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
#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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

// Check that we vectorize reductions with subi.
//
// CHECK-ON-LABEL:   func.func @sparse_reduction_subi(
// CHECK-ON-SAME:      %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:      %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-ON-DAG:       %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:       %[[VAL_4:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:           %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:           %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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
// CHECK-OFF-SAME:      %[[VAL_0:.*]]: tensor<i32>,
// CHECK-OFF-SAME:      %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-OFF-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:           %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:           %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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
#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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

// Check that we vectorize reductions with xor.

// CHECK-ON-LABEL: func.func @sparse_reduction_xor(
// CHECK-ON-SAME: %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME: %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-ON-DAG:  %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:  %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON-DAG:  %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:  %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:  %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:  %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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
// CHECK-OFF-SAME:  %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-OFF-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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

// Check that we vectorize reductions with addi.

// CHECK-ON-LABEL: func.func @sparse_reduction_addi(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<i32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-ON-DAG:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:   %[[VAL_3:.*]] = arith.constant dense<0> : vector<8xi32>
// CHECK-ON-DAG:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xi32, #sparse{{[0-9]*}}>) -> tensor<i32> {
// CHECK-OFF-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xi32, #sparse{{[0-9]*}}> to memref<?xi32>
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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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

// Check that we vectorize reductions with subf.

// CHECK-ON-LABEL: func.func @sparse_reduction_subf(
// CHECK-ON-SAME:   %[[VAL_0:.*]]: tensor<f32>,
// CHECK-ON-SAME:   %[[VAL_1:.*]]: tensor<?xf32, #sparse{{[0-9]*}}>) -> tensor<f32> {
// CHECK-ON-DAG:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:   %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-ON-DAG:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
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
// CHECK-OFF-SAME:   %[[VAL_1:.*]]: tensor<?xf32, #sparse{{[0-9]*}}>) -> tensor<f32> {
// CHECK-OFF-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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

// Check that we vectorize reductions with addf.

// CHECK-ON-LABEL: func.func @sparse_reduction_addf(
// CHECK-ON-SAME:  %[[VAL_0:.*]]: tensor<f32>,
// CHECK-ON-SAME:  %[[VAL_1:.*]]: tensor<?xf32, #sparse{{[0-9]*}}>) -> tensor<f32> {
// CHECK-ON-DAG:   %[[VAL_2:.*]] = arith.constant 8 : index
// CHECK-ON-DAG:   %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-ON-DAG:   %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-ON-DAG:   %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-ON:   %[[VAL_6:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-ON:   %[[VAL_7:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
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
// CHECK-OFF-SAME:  %[[VAL_0:.*]]: tensor<f32>,
// CHECK-OFF-SAME:  %[[VAL_1:.*]]: tensor<?xf32, #sparse{{[0-9]*}}>) -> tensor<f32> {
// CHECK-OFF-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-OFF-DAG:   %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-OFF:   %[[VAL_4:.*]] = sparse_tensor.positions %[[VAL_1]] {level = 0 : index} : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xindex>
// CHECK-OFF:   %[[VAL_5:.*]] = sparse_tensor.values %[[VAL_1]] : tensor<?xf32, #sparse{{[0-9]*}}> to memref<?xf32>
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

#SparseVector = #sparse_tensor.encoding<{map = (d0) -> (d0 : compressed)}>

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
