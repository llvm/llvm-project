// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @load_1D_vector(%source: memref<8x16x32xf32>, %offset: index) -> vector<8xf32> {
  %0 = vector.load %source[%offset, %offset, %offset]
    : memref<8x16x32xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @load_1D_vector(
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_2D_vector(%source: memref<8x16x32xf32>,
    %offset: index) -> vector<8x16xf32> {
  %0 = vector.load %source[%offset, %offset, %offset]
    : memref<8x16x32xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_2D_vector(
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_dynamic_source(%source: memref<?x?x?xf32>,
    %offset: index) -> vector<8x16xf32> {
  %0 = vector.load %source[%offset, %offset, %offset]
    : memref<?x?x?xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_dynamic_source(
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
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_out_of_bounds(%source: memref<7x15xf32>,
    %offset: index) -> vector<8x16xf32> {
  %0 = vector.load %source[%offset, %offset]
    : memref<7x15xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_out_of_bounds(
// CHECK-SAME:  %[[SRC:.+]]: memref<7x15xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<7x15xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @no_load_high_dim_vector(%source: memref<16x32x64xf32>,
    %offset: index) -> vector<8x16x32xf32> {
  %0 = vector.load %source[%offset, %offset, %offset]
    : memref<16x32x64xf32>, vector<8x16x32xf32>
  return %0 : vector<8x16x32xf32>
}

// CHECK-LABEL: @no_load_high_dim_vector(
// CHECK:       vector.load

// -----

func.func @no_load_zero_dim_vector(%source: memref<64xf32>,
    %offset: index) -> vector<f32> {
  %0 = vector.load %source[%offset]
    : memref<64xf32>, vector<f32>
  return %0 : vector<f32>
}

// CHECK-LABEL: @no_load_zero_dim_vector(
// CHECK:       vector.load
