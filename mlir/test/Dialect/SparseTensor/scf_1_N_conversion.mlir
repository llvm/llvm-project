// RUN: mlir-opt %s -sparse-tensor-codegen -cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK-LABEL:   func.func @for(
// CHECK-SAME:                   %[[VAL_1:.*0]]: memref<?xindex>,
// CHECK-SAME:                   %[[VAL_2:.*1]]: memref<?xindex>,
// CHECK-SAME:                   %[[VAL_3:.*2]]: memref<?xf32>,
// CHECK-SAME:                   %[[VAL_4:.*3]]: !sparse_tensor.storage_specifier
// CHECK-SAME:                   %[[VAL_5:.*4]]: index,
// CHECK-SAME:                   %[[VAL_6:.*5]]: index,
// CHECK-SAME:                   %[[VAL_7:.*6]]: index) -> (memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier
// CHECK:           %[[VAL_8:.*]]:4 = scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] iter_args(
// CHECK-SAME:        %[[VAL_11:.*]] = %[[VAL_1]],
// CHECK-SAME:        %[[VAL_12:.*]] = %[[VAL_2]],
// CHECK-SAME:        %[[VAL_13:.*]] = %[[VAL_3]],
// CHECK-SAME:        %[[VAL_14:.*]] = %[[VAL_4]])
// CHECK:             scf.yield %[[VAL_11]], %[[VAL_12]], %[[VAL_13]], %[[VAL_14]] :
// CHECK:           }
// CHECK:           return %[[VAL_8]]#0, %[[VAL_8]]#1, %[[VAL_8]]#2, %[[VAL_8]]#3
func.func @for(%in: tensor<1024xf32, #SparseVector>,
               %lb: index, %ub: index, %step: index) -> tensor<1024xf32, #SparseVector> {
  %1 = scf.for %i = %lb to %ub step %step iter_args(%vin = %in)
     -> tensor<1024xf32, #SparseVector> {
    scf.yield %vin : tensor<1024xf32, #SparseVector>
  }
  return %1 : tensor<1024xf32, #SparseVector>
}

// CHECK-LABEL:   func.func @if(
// CHECK-SAME:                  %[[VAL_1:.*0]]: memref<?xindex>,
// CHECK-SAME:                  %[[VAL_2:.*1]]: memref<?xindex>,
// CHECK-SAME:                  %[[VAL_3:.*2]]: memref<?xf32>,
// CHECK-SAME:                  %[[VAL_4:.*3]]: !sparse_tensor.storage_specifier
// CHECK-SAME:                  %[[VAL_6:.*4]]: memref<?xindex>,
// CHECK-SAME:                  %[[VAL_7:.*5]]: memref<?xindex>,
// CHECK-SAME:                  %[[VAL_8:.*6]]: memref<?xf32>,
// CHECK-SAME:                  %[[VAL_9:.*7]]: !sparse_tensor.storage_specifier
// CHECK-SAME:                  %[[VAL_10:.*]]: i1)
// CHECK:           %[[VAL_11:.*]]:4 = scf.if %[[VAL_10]]
// CHECK:             scf.yield %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]]
// CHECK:           } else {
// CHECK:             scf.yield %[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]
// CHECK:           }
// CHECK:           return %[[VAL_11]]#0, %[[VAL_11]]#1, %[[VAL_11]]#2, %[[VAL_11]]#3 :
// CHECK-SAME:        memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier
func.func @if(%t: tensor<1024xf32, #SparseVector>,
              %f: tensor<1024xf32, #SparseVector>,
              %c: i1) -> tensor<1024xf32, #SparseVector> {
  %1 = scf.if %c -> tensor<1024xf32, #SparseVector> {
    scf.yield %t : tensor<1024xf32, #SparseVector>
  } else {
    scf.yield %f : tensor<1024xf32, #SparseVector>
  }
  return %1 : tensor<1024xf32, #SparseVector>
}


// CHECK-LABEL:   func.func @while(
// CHECK-SAME:                     %[[VAL_1:.*0]]: memref<?xindex>,
// CHECK-SAME:                     %[[VAL_2:.*1]]: memref<?xindex>,
// CHECK-SAME:                     %[[VAL_3:.*2]]: memref<?xf32>,
// CHECK-SAME:                     %[[VAL_4:.*3]]: !sparse_tensor.storage_specifier
// CHECK-SAME:                     %[[VAL_5:.*4]]: i1)
// CHECK:           %[[VAL_6:.*]]:4 = scf.while (
// CHECK-SAME:        %[[VAL_8:.*]] = %[[VAL_1]],
// CHECK-SAME:        %[[VAL_9:.*]] = %[[VAL_2]],
// CHECK-SAME:        %[[VAL_10:.*]] = %[[VAL_3]],
// CHECK-SAME:        %[[VAL_11:.*]] = %[[VAL_4]])
// CHECK:             scf.condition(%[[VAL_5]]) %[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_13:.*5]]: memref<?xindex>,
// CHECK-SAME:           %[[VAL_14:.*6]]: memref<?xindex>,
// CHECK-SAME:           %[[VAL_15:.*7]]: memref<?xf32>,
// CHECK-SAME:           %[[VAL_16:.*8]]: !sparse_tensor.storage_specifier
// CHECK:             scf.yield %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]]
// CHECK:           }
// CHECK:           return %[[VAL_6]]#0, %[[VAL_6]]#1, %[[VAL_6]]#2, %[[VAL_6]]#3 :
// CHECK-SAME:        memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier
func.func @while(%arg0: tensor<1024xf32, #SparseVector>, %c: i1) -> tensor<1024xf32, #SparseVector> {
  %0 = scf.while (%in = %arg0) : (tensor<1024xf32, #SparseVector>) -> tensor<1024xf32, #SparseVector> {
    scf.condition(%c) %in : tensor<1024xf32, #SparseVector>
  } do {
  ^bb0(%arg1: tensor<1024xf32, #SparseVector>):
    scf.yield %arg1 : tensor<1024xf32, #SparseVector>
  }
  return %0: tensor<1024xf32, #SparseVector>
}
