// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// This file contains some tests of folding/canonicalizing vector.extract

//-----------------------------------------------------------------------------
// [Pattern: ExtractOpFromLoad]
//-----------------------------------------------------------------------------

// CHECK-LABEL: @extract_load_scalar
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @extract_load_scalar(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_index
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xindex>, %[[ARG1:.*]]: index)
func.func @extract_load_index(%arg0: memref<?xindex>, %arg1: index) -> index {
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]]] : memref<?xindex>
// CHECK:   return %[[RES]] : index
  %0 = vector.load %arg0[%arg1] : memref<?xindex>, vector<4xindex>
  %1 = vector.extract %0[0] : index from vector<4xindex>
  return %1 : index
}

// CHECK-LABEL: @extract_load_scalar_non_zero_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @extract_load_scalar_non_zero_off(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[OFF]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_scalar_dyn_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_scalar_dyn_off(%arg0: memref<?xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[ARG2]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[OFF]]] : memref<?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[%arg2] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_vec_non_zero_off
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_vec_non_zero_off(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> vector<4xf32> {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG1]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[OFF]], %[[ARG2]]] : memref<?x?xf32>, vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = vector.load %arg0[%arg1, %arg2] : memref<?x?xf32>, vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @extract_load_scalar_non_zero_off_2d_src_memref
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_load_scalar_non_zero_off_2d_src_memref(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> f32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG2]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = memref.load %[[ARG0]][%[[ARG1]], %[[OFF]]] : memref<?x?xf32>
// CHECK:   return %[[RES]] : f32
  %0 = vector.load %arg0[%arg1, %arg2] : memref<?x?xf32>, vector<4xf32>
  %1 = vector.extract %0[1] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract_load_vec_high_rank
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?x?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index)
func.func @extract_load_vec_high_rank(%arg0: memref<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index) -> vector<4xf32> {
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[OFF:.*]] = arith.addi %[[ARG2]], %[[C1]] overflow<nsw> : index
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]], %[[OFF]], %[[ARG3]]] : memref<?x?x?xf32>, vector<4xf32>
// CHECK:   return %[[RES]] : vector<4xf32>
  %0 = vector.load %arg0[%arg1, %arg2, %arg3] : memref<?x?x?xf32>, vector<2x4xf32>
  %1 = vector.extract %0[1] : vector<4xf32> from vector<2x4xf32>
  return %1 : vector<4xf32>
}

// CHECK-LABEL: @negative_extract_load_scalar_from_memref_of_vec
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xvector<4xf32>>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalar_from_memref_of_vec(%arg0: memref<?xvector<4xf32>>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xvector<4xf32>>, vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xvector<4xf32>>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1 : f32
}

// CHECK-LABEL: @negative_extract_load_scalar_from_memref_of_i1
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xi1>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalar_from_memref_of_i1(%arg0: memref<?xi1>, %arg1: index) -> i1 {
// Subbyte types are tricky, ignore them for now.
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xi1>, vector<8xi1>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : i1 from vector<8xi1>
// CHECK:   return %[[EXT]] : i1
  %0 = vector.load %arg0[%arg1] : memref<?xi1>, vector<8xi1>
  %1 = vector.extract %0[0] : i1 from vector<8xi1>
  return %1 : i1
}

// CHECK-LABEL: @negative_extract_load_no_single_use
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_no_single_use(%arg0: memref<?xf32>, %arg1: index) -> (f32, vector<4xf32>) {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<4xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<4xf32>
// CHECK:   return %[[EXT]], %[[RES]] : f32, vector<4xf32>
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<4xf32>
  %1 = vector.extract %0[0] : f32 from vector<4xf32>
  return %1, %0 : f32, vector<4xf32>
}

// CHECK-LABEL: @negative_extract_load_scalable
//  CHECK-SAME:   (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index)
func.func @negative_extract_load_scalable(%arg0: memref<?xf32>, %arg1: index) -> f32 {
// CHECK:   %[[RES:.*]] = vector.load %[[ARG0]][%[[ARG1]]] : memref<?xf32>, vector<[1]xf32>
// CHECK:   %[[EXT:.*]] = vector.extract %[[RES]][0] : f32 from vector<[1]xf32>
// CHECK:   return %[[EXT]] : f32
  %0 = vector.load %arg0[%arg1] : memref<?xf32>, vector<[1]xf32>
  %1 = vector.extract %0[0] : f32 from vector<[1]xf32>
  return %1 : f32
}
