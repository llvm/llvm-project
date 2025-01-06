// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @store_1D_vector(%vec: vector<8xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.store %vec, %source[%offset, %offset, %offset]
    : memref<8x16x32xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: @store_1D_vector(
// CHECK-SAME:  %[[VEC:.+]]: vector<8xf32>,
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8xf32>

// -----

func.func @store_2D_vector(%vec: vector<8x16xf32>,
    %source: memref<8x16x32xf32>, %offset: index) {
  vector.store %vec, %source[%offset, %offset, %offset]
    : memref<8x16x32xf32>, vector<8x16xf32>
  return
}

// CHECK-LABEL: @store_2D_vector(
// CHECK-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

// -----

func.func @store_dynamic_source(%vec: vector<8x16xf32>,
    %source: memref<?x?x?xf32>, %offset: index) {
  vector.store %vec, %source[%offset, %offset, %offset]
    : memref<?x?x?xf32>, vector<8x16xf32>
  return
}

// CHECK-LABEL: @store_dynamic_source(
// CHECK-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[DIM_0:.+]] = memref.dim %[[SRC]], %[[C0]]
// CHECK-DAG:   %[[DIM_1:.+]] = memref.dim %[[SRC]], %[[C1]]
// CHECK-DAG:   %[[DIM_2:.+]] = memref.dim %[[SRC]], %[[C2]]
// CHECK:       %[[DIM_0_STRIDE:.+]] = arith.muli %[[DIM_2]], %[[DIM_1]]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    [%[[DIM_0]], %[[DIM_1]], %[[DIM_2]]], [%[[DIM_0_STRIDE]], %[[DIM_2]], 1]
// CHECK-SAME:    memref<?x?x?xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

// -----

func.func @store_out_of_bounds(%vec: vector<8x16xf32>,
    %source: memref<7x64xf32>, %offset: index) {
  vector.store %vec, %source[%offset, %offset]
    : memref<7x64xf32>, vector<8x16xf32>
  return
}

// CHECK-LABEL:   @store_out_of_bounds(
// CHECK-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[SRC:.+]]: memref<7x64xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<7x64xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]] : vector<8x16xf32>

// -----

func.func @no_store_high_dim_vector(%vec: vector<8x16x32xf32>,
    %source: memref<16x32x64xf32>, %offset: index) {
  vector.store %vec, %source[%offset, %offset, %offset]
    : memref<16x32x64xf32>, vector<8x16x32xf32>
  return
}

// CHECK-LABEL: @no_store_high_dim_vector(
// CHECK:       vector.store

// -----

func.func @no_store_zero_dim_vector(%vec: vector<f32>,
    %source: memref<64xf32>, %offset: index) {
  vector.store %vec, %source[%offset]
    : memref<64xf32>, vector<f32>
  return
}

// CHECK-LABEL: @no_store_zero_dim_vector(
// CHECK:       vector.store
