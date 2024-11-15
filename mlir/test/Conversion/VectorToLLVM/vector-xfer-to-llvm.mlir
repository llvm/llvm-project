// RUN: mlir-opt %s -convert-vector-to-llvm -split-input-file | FileCheck '-D$IDX_TYPE=i32' %s
// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=0' | FileCheck '-D$IDX_TYPE=i64' %s

func.func @transfer_read_write_1d(%A : memref<?xf32>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1], %[[CST]] : memref<?xf32>, vector<17xf32>
// CHECK: vector.transfer_write %[[TRANSFER_READ]], %arg0[%arg1] : vector<17xf32>, memref<?xf32>

// -----

func.func @transfer_read_write_1d_scalable(%A : memref<?xf32>, %base: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<[17]xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<[17]xf32>, memref<?xf32>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d_scalable
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1], %[[CST]] : memref<?xf32>, vector<[17]xf32>
// CHECK: vector.transfer_write %[[TRANSFER_READ]], %arg0[%arg1] : vector<[17]xf32>, memref<?xf32>

// -----

func.func @transfer_read_write_index_1d(%A : memref<?xindex>, %base: index) -> vector<17xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<17xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xindex>, memref<?xindex>
  return %f: vector<17xindex>
}
// CHECK-LABEL: func @transfer_read_write_index_1d
// CHECK: %[[CST:.*]] = arith.constant 7 : index
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1], %[[CST]] : memref<?xindex>, vector<17xindex>
// CHECK: vector.transfer_write %[[TRANSFER_READ]], %arg0[%arg1] : vector<17xindex>, memref<?xindex>

func.func @transfer_read_write_index_1d_scalable(%A : memref<?xindex>, %base: index) -> vector<[17]xindex> {
  %f7 = arith.constant 7: index
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xindex>, vector<[17]xindex>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<[17]xindex>, memref<?xindex>
  return %f: vector<[17]xindex>
}
// CHECK-LABEL: func @transfer_read_write_index_1d_scalable
// CHECK: %[[CST:.*]] = arith.constant 7 : index
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1], %[[CST]] : memref<?xindex>, vector<[17]xindex>
// CHECK: vector.transfer_write %[[TRANSFER_READ]], %arg0[%arg1] : vector<[17]xindex>, memref<?xindex>

// -----

func.func @transfer_read_2d_to_1d(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<17xf32>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1, %arg2], %[[CST]] : memref<?x?xf32>, vector<17xf32>

func.func @transfer_read_2d_to_1d_scalable(%A : memref<?x?xf32>, %base0: index, %base1: index) -> vector<[17]xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base0, %base1], %f7
      {permutation_map = affine_map<(d0, d1) -> (d1)>} :
    memref<?x?xf32>, vector<[17]xf32>
  return %f: vector<[17]xf32>
}
// CHECK-LABEL: func @transfer_read_2d_to_1d_scalable
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1, %arg2], %[[CST]] : memref<?x?xf32>, vector<[17]xf32>
// CHECK: return %[[TRANSFER_READ]] : vector<[17]xf32>

// -----

func.func @transfer_read_write_1d_non_zero_addrspace(%A : memref<?xf32, 3>, %base: index) -> vector<17xf32> {
  %f7 = arith.constant 7.0: f32
  %f = vector.transfer_read %A[%base], %f7
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32, 3>, vector<17xf32>
  vector.transfer_write %f, %A[%base]
      {permutation_map = affine_map<(d0) -> (d0)>} :
    vector<17xf32>, memref<?xf32, 3>
  return %f: vector<17xf32>
}
// CHECK-LABEL: func @transfer_read_write_1d_non_zero_addrspace
// CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: %[[TRANSFER_READ:.*]] = vector.transfer_read %arg0[%arg1], %[[CST]] : memref<?xf32, 3>, vector<17xf32>
// CHECK: vector.transfer_write %[[TRANSFER_READ]], %arg0[%arg1] : vector<17xf32>, memref<?xf32, 3>
// CHECK: return %[[TRANSFER_READ]] : 
