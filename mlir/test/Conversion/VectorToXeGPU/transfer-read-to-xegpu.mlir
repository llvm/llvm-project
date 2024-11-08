// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @load_1D_vector(%source: memref<8x16x32xf32>, %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true]} : memref<8x16x32xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @load_1D_vector(
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_2D_vector(%source: memref<8x16x32xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true, true]} : memref<8x16x32xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_2D_vector(
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[SRC]][%[[OFFSET]], %[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<8x16x32xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_zero_pad_out_of_bounds(%source: memref<32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {in_bounds = [false, true]} : memref<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_zero_pad_out_of_bounds(
// CHECK-SAME:  %[[SRC:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<8x16xf32,
// CHECK-SAME:    boundary_check = true
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_transposed(%source: memref<32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_transposed(
// CHECK-SAME:  %[[SRC:.+]]: memref<32x64xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[SRC]][%[[OFFSET]], %[[OFFSET]]]
// CHECK-SAME:    memref<32x64xf32> -> !xegpu.tensor_desc<16x8xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]] <{transpose = array<i64: 1, 0>}>
// CHECK-SAME:    -> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_dynamic_source(%source: memref<?x?x?xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {in_bounds = [true, true]} : memref<?x?x?xf32>, vector<8x16xf32>
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
// CHECK-SAME:    memref<?x?x?xf32> -> !xegpu.tensor_desc<8x16xf32
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @no_load_out_of_bounds_non_zero_pad(%source: memref<32x64xf32>,
    %offset: index, %arg2: index, %pad: f32) -> (vector<8x16xf32>, vector<8x16xf32>) {
  %c1 = arith.constant 1.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2], %c1
    {in_bounds = [true, false]} : memref<32x64xf32>, vector<8x16xf32>
  %1 = vector.transfer_read %source[%arg2, %offset], %pad
    {in_bounds = [false, true]} : memref<32x64xf32>, vector<8x16xf32>
  return %0, %1 : vector<8x16xf32>, vector<8x16xf32>
}

// CHECK-LABEL:   @no_load_out_of_bounds_non_zero_pad(
// CHECK-COUNT-2: vector.transfer_read

// -----

func.func @no_load_masked(%source : memref<4xf32>,
    %offset : index) -> vector<4xf32> {
  %c0 = arith.constant 0.0 : f32
  %mask = arith.constant dense<[0, 1, 0, 1]> : vector<4xi1>
  %0 = vector.transfer_read %source[%offset], %c0, %mask
    {in_bounds = [true]} : memref<4xf32>, vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: @no_load_masked(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_tensor(%source: tensor<32x64xf32>,
    %offset: index, %arg2: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2], %c0
    {in_bounds = [true, true]} : tensor<32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @no_load_tensor(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_high_dim_vector(%source: memref<16x32x64xf32>,
    %offset: index, %arg2: index) -> vector<8x16x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %arg2, %offset], %c0
    {in_bounds = [true, true, true]} : memref<16x32x64xf32>, vector<8x16x32xf32>
  return %0 : vector<8x16x32xf32>
}

// CHECK-LABEL: @no_load_high_dim_vector(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_non_unit_inner_stride(
    %source: memref<32xf32, strided<[?], offset: ?>>,
    %offset: index) -> vector<8xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset], %c0 {in_bounds = [true]}
    : memref<32xf32, strided<[?], offset: ?>>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @no_load_non_unit_inner_stride(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_unsupported_map(%source: memref<16x32x64xf32>,
    %offset: index) -> vector<8x16xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %source[%offset, %offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2)>,
    in_bounds = [true, true]} : memref<16x32x64xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @no_load_unsupported_map(
// CHECK:       vector.transfer_read

// -----

func.func @no_load_transpose_unsupported_data_type(%source: memref<32x64xf16>,
    %offset: index) -> vector<8x16xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = vector.transfer_read %source[%offset, %offset], %c0
    {permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
    in_bounds = [true, true]} : memref<32x64xf16>, vector<8x16xf16>
  return %0 : vector<8x16xf16>
}

// CHECK-LABEL: @no_load_transpose_unsupported_data_type(
// CHECK:       vector.transfer_read
