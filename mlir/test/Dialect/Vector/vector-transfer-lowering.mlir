// RUN: mlir-opt %s -test-vector-transfer-lowering-patterns -split-input-file | FileCheck %s

// transfer_read/write are lowered to vector.load/store
// CHECK-LABEL:   func @transfer_to_load(
// CHECK-SAME:                                %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                                %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      vector.store  %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func @transfer_to_load(%mem : memref<8x8xf32>, %i : index) -> vector<4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false]} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [false]} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// n-D results are also supported.
// CHECK-LABEL:   func @transfer_2D(
// CHECK-SAME:                           %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                           %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      vector.store %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func @transfer_2D(%mem : memref<8x8xf32>, %i : index) -> vector<2x4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false, false]} : memref<8x8xf32>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [false, false]} : vector<2x4xf32>, memref<8x8xf32>
  return %res : vector<2x4xf32>
}

// -----

// Vector element types are supported when the result has the same type.
// CHECK-LABEL:   func @transfer_vector_element(
// CHECK-SAME:                           %[[MEM:.*]]: memref<8x8xvector<2x4xf32>>,
// CHECK-SAME:                           %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[RES:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
// CHECK-NEXT:      vector.store %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func @transfer_vector_element(%mem : memref<8x8xvector<2x4xf32>>, %i : index) -> vector<2x4xf32> {
  %cf0 = constant dense<0.0> : vector<2x4xf32>
  %res = vector.transfer_read %mem[%i, %i], %cf0 : memref<8x8xvector<2x4xf32>>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%i, %i] : vector<2x4xf32>, memref<8x8xvector<2x4xf32>>
  return %res : vector<2x4xf32>
}

// -----

// TODO: Vector element types are not supported yet when the result has a
// different type.
// CHECK-LABEL:   func @transfer_vector_element_different_types(
// CHECK-SAME:                           %[[MEM:.*]]: memref<8x8xvector<2x4xf32>>,
// CHECK-SAME:                           %[[IDX:.*]]: index) -> vector<1x2x4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = constant dense<0.000000e+00> : vector<2x4xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {masked = [false]} : memref<8x8xvector<2x4xf32>>, vector<1x2x4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES:.*]], %[[MEM]][%[[IDX]], %[[IDX]]] {masked = [false]} : vector<1x2x4xf32>, memref<8x8xvector<2x4xf32>>
// CHECK-NEXT:      return %[[RES]] : vector<1x2x4xf32>
// CHECK-NEXT:    }

func @transfer_vector_element_different_types(%mem : memref<8x8xvector<2x4xf32>>, %i : index) -> vector<1x2x4xf32> {
  %cf0 = constant dense<0.0> : vector<2x4xf32>
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false]} : memref<8x8xvector<2x4xf32>>, vector<1x2x4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [false]} : vector<1x2x4xf32>, memref<8x8xvector<2x4xf32>>
  return %res : vector<1x2x4xf32>
}

// -----

// TODO: transfer_read/write cannot be lowered because there is an unmasked
// dimension.
// CHECK-LABEL:   func @transfer_2D_masked(
// CHECK-SAME:                                  %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                                  %[[IDX:.*]]: index) -> vector<2x4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {masked = [false, true]} : memref<8x8xf32>, vector<2x4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] {masked = [true, false]} : vector<2x4xf32>, memref<8x8xf32>
// CHECK-NEXT:      return %[[RES]] : vector<2x4xf32>
// CHECK-NEXT:    }

func @transfer_2D_masked(%mem : memref<8x8xf32>, %i : index) -> vector<2x4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false, true]} : memref<8x8xf32>, vector<2x4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [true, false]} : vector<2x4xf32>, memref<8x8xf32>
  return %res : vector<2x4xf32>
}

// -----

// TODO: transfer_read/write cannot be lowered because they are masked.
// CHECK-LABEL:   func @transfer_masked(
// CHECK-SAME:                               %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                               %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] : vector<4xf32>, memref<8x8xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func @transfer_masked(%mem : memref<8x8xf32>, %i : index) -> vector<4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%i, %i] : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// TODO: transfer_read/write cannot be lowered to vector.load/store because the
// memref has a non-default layout.
// CHECK-LABEL:   func @transfer_nondefault_layout(
// CHECK-SAME:                                          %[[MEM:.*]]: memref<8x8xf32, #{{.*}}>,
// CHECK-SAME:                                          %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {masked = [false]} : memref<8x8xf32, #{{.*}}>, vector<4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] {masked = [false]} : vector<4xf32>, memref<8x8xf32, #{{.*}}>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

#layout = affine_map<(d0, d1) -> (d0*16 + d1)>
func @transfer_nondefault_layout(%mem : memref<8x8xf32, #layout>, %i : index) -> vector<4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false]} : memref<8x8xf32, #layout>, vector<4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [false]} : vector<4xf32>, memref<8x8xf32, #layout>
  return %res : vector<4xf32>
}

// -----

// TODO: transfer_read/write cannot be lowered to vector.load/store yet when the
// permutation map is not the minor identity map (up to broadcasting).
// CHECK-LABEL:   func @transfer_perm_map(
// CHECK-SAME:                                 %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                                 %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[CF0:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[RES:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]]], %[[CF0]] {masked = [false], permutation_map = #{{.*}}} : memref<8x8xf32>, vector<4xf32>
// CHECK-NEXT:      vector.transfer_write %[[RES]], %[[MEM]][%[[IDX]], %[[IDX]]] {masked = [false], permutation_map = #{{.*}}} : vector<4xf32>, memref<8x8xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

func @transfer_perm_map(%mem : memref<8x8xf32>, %i : index) -> vector<4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<8x8xf32>, vector<4xf32>
  vector.transfer_write %res, %mem[%i, %i] {masked = [false], permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<4xf32>, memref<8x8xf32>
  return %res : vector<4xf32>
}

// -----

// Lowering of transfer_read with broadcasting is supported (note that a `load`
// is generated instead of a `vector.load`).
// CHECK-LABEL:   func @transfer_broadcasting(
// CHECK-SAME:                                %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                                %[[IDX:.*]]: index) -> vector<4xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : f32 to vector<4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4xf32>
// CHECK-NEXT:    }

#broadcast = affine_map<(d0, d1) -> (0)>
func @transfer_broadcasting(%mem : memref<8x8xf32>, %i : index) -> vector<4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false], permutation_map = #broadcast} : memref<8x8xf32>, vector<4xf32>
  return %res : vector<4xf32>
}

// -----

// An example with two broadcasted dimensions.
// CHECK-LABEL:   func @transfer_broadcasting_2D(
// CHECK-SAME:                                %[[MEM:.*]]: memref<8x8xf32>,
// CHECK-SAME:                                %[[IDX:.*]]: index) -> vector<4x4xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = load %[[MEM]][%[[IDX]], %[[IDX]]] : memref<8x8xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : f32 to vector<4x4xf32>
// CHECK-NEXT:      return %[[RES]] : vector<4x4xf32>
// CHECK-NEXT:    }

#broadcast = affine_map<(d0, d1) -> (0, 0)>
func @transfer_broadcasting_2D(%mem : memref<8x8xf32>, %i : index) -> vector<4x4xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i], %cf0 {masked = [false, false], permutation_map = #broadcast} : memref<8x8xf32>, vector<4x4xf32>
  return %res : vector<4x4xf32>
}

// -----

// More complex broadcasting case (here a `vector.load` is generated).
// CHECK-LABEL:   func @transfer_broadcasting_complex(
// CHECK-SAME:                                %[[MEM:.*]]: memref<10x20x30x8x8xf32>,
// CHECK-SAME:                                %[[IDX:.*]]: index) -> vector<3x2x4x5xf32> {
// CHECK-NEXT:      %[[LOAD:.*]] = vector.load %[[MEM]][%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]] : memref<10x20x30x8x8xf32>, vector<3x1x1x5xf32>
// CHECK-NEXT:      %[[RES:.*]] = vector.broadcast %[[LOAD]] : vector<3x1x1x5xf32> to vector<3x2x4x5xf32>
// CHECK-NEXT:      return %[[RES]] : vector<3x2x4x5xf32>
// CHECK-NEXT:    }

#broadcast = affine_map<(d0, d1, d2, d3, d4) -> (d1, 0, 0, d4)>
func @transfer_broadcasting_complex(%mem : memref<10x20x30x8x8xf32>, %i : index) -> vector<3x2x4x5xf32> {
  %cf0 = constant 0.0 : f32
  %res = vector.transfer_read %mem[%i, %i, %i, %i, %i], %cf0 {masked = [false, false, false, false], permutation_map = #broadcast} : memref<10x20x30x8x8xf32>, vector<3x2x4x5xf32>
  return %res : vector<3x2x4x5xf32>
}
