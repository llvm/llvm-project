// RUN: mlir-opt %s -sparse-tensor-codegen -cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
// CHECK-LABEL:  func @for(
// CHECK-SAME:             %[[DIM_SIZE:.*0]]: memref<1xindex>,
// CHECK-SAME:             %[[MEM_SIZE:.*1]]: memref<3xindex>,
// CHECK-SAME:             %[[POINTER:.*2]]: memref<?xindex>,
// CHECK-SAME:             %[[INDICES:.*3]]: memref<?xindex>,
// CHECK-SAME:             %[[VALUE:.*4]]: memref<?xf32>,
// CHECK-SAME:             %[[LB:.*5]]: index,
// CHECK-SAME:             %[[UB:.*6]]: index,
// CHECK-SAME:             %[[STEP:.*7]]: index)
// CHECK:          %[[OUT:.*]]:5 = scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(
// CHECK-SAME:       %[[SIZE:.*]] = %[[DIM_SIZE]],
// CHECK-SAME:       %[[MEM:.*]] = %[[MEM_SIZE]],
// CHECK-SAME:       %[[PTR:.*]] = %[[POINTER]],
// CHECK-SAME:       %[[IDX:.*]] = %[[INDICES]],
// CHECK-SAME:       %[[VAL:.*]] = %[[VALUE]])
// CHECK:            scf.yield %[[SIZE]], %[[MEM]], %[[PTR]], %[[IDX]], %[[VAL]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:          }
// CHECK:          return %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2, %[[OUT]]#3, %[[OUT]]#4 : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
func.func @for(%in: tensor<1024xf32, #SparseVector>,
               %lb: index, %ub: index, %step: index) -> tensor<1024xf32, #SparseVector> {
  %1 = scf.for %i = %lb to %ub step %step iter_args(%vin = %in)
     -> tensor<1024xf32, #SparseVector> {
    scf.yield %vin : tensor<1024xf32, #SparseVector>
  }
  return %1 : tensor<1024xf32, #SparseVector>
}


// CHECK-LABEL:  func @if(
//  CHECK-SAME:          %[[DIM_SIZE:.*0]]: memref<1xindex>,
//  CHECK-SAME:          %[[MEM_SIZE:.*1]]: memref<3xindex>,
//  CHECK-SAME:          %[[POINTER:.*2]]: memref<?xindex>,
//  CHECK-SAME:          %[[INDICES:.*3]]: memref<?xindex>,
//  CHECK-SAME:          %[[VALUE:.*4]]: memref<?xf32>,
//  CHECK-SAME:          %[[DIM_SIZE_1:.*5]]: memref<1xindex>,
//  CHECK-SAME:          %[[MEM_SIZE_1:.*6]]: memref<3xindex>,
//  CHECK-SAME:          %[[POINTER_1:.*7]]: memref<?xindex>,
//  CHECK-SAME:          %[[INDICES_1:.*8]]: memref<?xindex>,
//  CHECK-SAME:          %[[VALUE_1:.*9]]: memref<?xf32>,
//  CHECK-SAME:          %[[I1:.*10]]: i1) ->
//  CHECK-SAME:          (memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:  %[[SV:.*]]:5 = scf.if %[[I1]] -> (memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:    scf.yield %[[DIM_SIZE]], %[[MEM_SIZE]], %[[POINTER]], %[[INDICES]], %[[VALUE]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[DIM_SIZE_1]], %[[MEM_SIZE_1]], %[[POINTER_1]], %[[INDICES_1]], %[[VALUE_1]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  }
//       CHECK:  return %[[SV]]#0, %[[SV]]#1, %[[SV]]#2, %[[SV]]#3, %[[SV]]#4 : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
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

// CHECK-LABEL:  func @while(
//  CHECK-SAME:              %[[DIM_SIZE:.*0]]: memref<1xindex>,
//  CHECK-SAME:              %[[MEM_SIZE:.*1]]: memref<3xindex>,
//  CHECK-SAME:              %[[POINTER:.*2]]: memref<?xindex>,
//  CHECK-SAME:              %[[INDICES:.*3]]: memref<?xindex>,
//  CHECK-SAME:              %[[VALUE:.*4]]: memref<?xf32>,
//  CHECK-SAME:              %[[I1:.*5]]: i1) ->
//  CHECK-SAME:              (memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:  %[[SV:.*]]:5 = scf.while (
//  CHECK-SAME:              %[[TMP_DIM:.*]] = %[[DIM_SIZE]],
//  CHECK-SAME:              %[[TMP_MEM:.*]] = %[[MEM_SIZE]],
//  CHECK-SAME:              %[[TMP_PTR:.*]] = %[[POINTER]],
//  CHECK-SAME:              %[[TMP_IND:.*]] = %[[INDICES]],
//  CHECK-SAME:              %[[TMP_VAL:.*]] = %[[VALUE]])
//       CHECK:    scf.condition(%[[I1]]) %[[TMP_DIM]], %[[TMP_MEM]], %[[TMP_PTR]], %[[TMP_IND]], %[[TMP_VAL]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  } do {
//       CHECK:  ^bb0(%[[TMP_DIM]]: memref<1xindex>, %[[TMP_MEM]]: memref<3xindex>, %[[TMP_PTR]]: memref<?xindex>, %[[TMP_IND]]: memref<?xindex>, %[[TMP_VAL]]: memref<?xf32>):
//       CHECK:    scf.yield %[[TMP_DIM]], %[[TMP_MEM]], %[[TMP_PTR]], %[[TMP_IND]], %[[TMP_VAL]] : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  }
//       CHECK:  return %[[SV]]#0, %[[SV]]#1, %[[SV]]#2, %[[SV]]#3, %[[SV]]#4 : memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
func.func @while(%arg0: tensor<1024xf32, #SparseVector>, %c: i1) -> tensor<1024xf32, #SparseVector> {
  %0 = scf.while (%in = %arg0) : (tensor<1024xf32, #SparseVector>) -> tensor<1024xf32, #SparseVector> {
    scf.condition(%c) %in : tensor<1024xf32, #SparseVector>
  } do {
  ^bb0(%arg1: tensor<1024xf32, #SparseVector>):
    scf.yield %arg1 : tensor<1024xf32, #SparseVector>
  }
  return %0: tensor<1024xf32, #SparseVector>
}
