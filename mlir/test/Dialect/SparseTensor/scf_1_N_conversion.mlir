// RUN: mlir-opt %s -sparse-tensor-codegen -cse | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>
// CHECK-LABEL:  func @for(
// CHECK-SAME:             %[[DIM_SIZE:.*0]]: memref<1xindex>,
// CHECK-SAME:             %[[DIM_CURSOR:.*1]]: memref<1xindex>,
// CHECK-SAME:             %[[MEM_SIZE:.*2]]: memref<3xindex>,
// CHECK-SAME:             %[[POINTER:.*3]]: memref<?xindex>,
// CHECK-SAME:             %[[INDICES:.*4]]: memref<?xindex>,
// CHECK-SAME:             %[[VALUE:.*5]]: memref<?xf32>,
// CHECK-SAME:             %[[LB:.*6]]: index,
// CHECK-SAME:             %[[UB:.*7]]: index,
// CHECK-SAME:             %[[STEP:.*8]]: index)
// CHECK:          %[[OUT:.*]]:6 = scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(
// CHECK-SAME:       %[[SIZE:.*]] = %[[DIM_SIZE]],
// CHECK-SAME:       %[[CUR:.*]] = %[[DIM_CURSOR]],
// CHECK-SAME:       %[[MEM:.*]] = %[[MEM_SIZE]],
// CHECK-SAME:       %[[PTR:.*]] = %[[POINTER]],
// CHECK-SAME:       %[[IDX:.*]] = %[[INDICES]],
// CHECK-SAME:       %[[VAL:.*]] = %[[VALUE]])
// CHECK:            scf.yield %[[SIZE]], %[[CUR]], %[[MEM]], %[[PTR]], %[[IDX]], %[[VAL]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
// CHECK:          }
// CHECK:          return %[[OUT]]#0, %[[OUT]]#1, %[[OUT]]#2, %[[OUT]]#3, %[[OUT]]#4, %[[OUT]]#5 : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
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
//  CHECK-SAME:          %[[DIM_CURSOR:.*1]]: memref<1xindex>,
//  CHECK-SAME:          %[[MEM_SIZE:.*2]]: memref<3xindex>,
//  CHECK-SAME:          %[[POINTER:.*3]]: memref<?xindex>,
//  CHECK-SAME:          %[[INDICES:.*4]]: memref<?xindex>,
//  CHECK-SAME:          %[[VALUE:.*5]]: memref<?xf32>,
//  CHECK-SAME:          %[[DIM_SIZE_1:.*6]]: memref<1xindex>,
//  CHECK-SAME:          %[[DIM_CURSOR_1:.*7]]: memref<1xindex>,
//  CHECK-SAME:          %[[MEM_SIZE_1:.*8]]: memref<3xindex>,
//  CHECK-SAME:          %[[POINTER_1:.*9]]: memref<?xindex>,
//  CHECK-SAME:          %[[INDICES_1:.*10]]: memref<?xindex>,
//  CHECK-SAME:          %[[VALUE_1:.*11]]: memref<?xf32>,
//  CHECK-SAME:          %[[TMP_arg12:.*12]]: i1) ->
//  CHECK-SAME:          (memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:  %[[SV:.*]]:6 = scf.if %[[TMP_arg12]] -> (memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:    scf.yield %[[DIM_SIZE]], %[[DIM_CURSOR]], %[[MEM_SIZE]], %[[POINTER]], %[[INDICES]], %[[VALUE]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  } else {
//       CHECK:    scf.yield %[[DIM_SIZE_1]], %[[DIM_CURSOR_1]], %[[MEM_SIZE_1]], %[[POINTER_1]], %[[INDICES_1]], %[[VALUE_1]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  }
//       CHECK:  return %[[SV]]#0, %[[SV]]#1, %[[SV]]#2, %[[SV]]#3, %[[SV]]#4, %[[SV]]#5 : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
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
//  CHECK-SAME:              %[[DIM_CURSOR:.*1]]: memref<1xindex>,
//  CHECK-SAME:              %[[MEM_SIZE:.*2]]: memref<3xindex>,
//  CHECK-SAME:              %[[POINTER:.*3]]: memref<?xindex>,
//  CHECK-SAME:              %[[INDICES:.*4]]: memref<?xindex>,
//  CHECK-SAME:              %[[VALUE:.*5]]: memref<?xf32>,
//  CHECK-SAME:              %[[TMP_arg6:.*6]]: i1) ->
//  CHECK-SAME:              (memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>) {
//       CHECK:  %[[SV:.*]]:6 = scf.while (
//  CHECK-SAME:              %[[TMP_arg7:.*]] = %[[DIM_SIZE]],
//  CHECK-SAME:              %[[TMP_arg8:.*]] = %[[DIM_CURSOR]],
//  CHECK-SAME:              %[[TMP_arg9:.*]] = %[[MEM_SIZE]],
//  CHECK-SAME:              %[[TMP_arg10:.*]] = %[[POINTER]],
//  CHECK-SAME:              %[[TMP_arg11:.*]] = %[[INDICES]],
//  CHECK-SAME:              %[[TMP_arg12:.*]] = %[[VALUE]]) 
//       CHECK:    scf.condition(%[[TMP_arg6]]) %[[TMP_arg7]], %[[TMP_arg8]], %[[TMP_arg9]], %[[TMP_arg10]], %[[TMP_arg11]], %[[TMP_arg12]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  } do {
//       CHECK:  ^bb0(%[[TMP_arg7]]: memref<1xindex>, %[[TMP_arg8]]: memref<1xindex>, %[[TMP_arg9]]: memref<3xindex>, %[[TMP_arg10]]: memref<?xindex>, %[[TMP_arg11]]: memref<?xindex>, %[[TMP_arg12]]: memref<?xf32>):
//       CHECK:    scf.yield %[[TMP_arg7]], %[[TMP_arg8]], %[[TMP_arg9]], %[[TMP_arg10]], %[[TMP_arg11]], %[[TMP_arg12]] : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
//       CHECK:  }
//       CHECK:  return %[[SV]]#0, %[[SV]]#1, %[[SV]]#2, %[[SV]]#3, %[[SV]]#4, %[[SV]]#5 : memref<1xindex>, memref<1xindex>, memref<3xindex>, memref<?xindex>, memref<?xindex>, memref<?xf32>
func.func @while(%arg0: tensor<1024xf32, #SparseVector>, %c: i1) -> tensor<1024xf32, #SparseVector> {
  %0 = scf.while (%arg4 = %arg0) : (tensor<1024xf32, #SparseVector>) -> tensor<1024xf32, #SparseVector> {
    scf.condition(%c) %arg4 : tensor<1024xf32, #SparseVector>
  } do {
  ^bb0(%arg7: tensor<1024xf32, #SparseVector>):
    scf.yield %arg7 : tensor<1024xf32, #SparseVector>
  }
  return %0: tensor<1024xf32, #SparseVector>
}
