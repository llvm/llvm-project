// RUN: mlir-opt %s -convert-vector-to-xegpu -split-input-file | FileCheck %s

func.func @load_1D_vector(%source: memref<8x16x32xf32>, %offset: index) -> vector<8xf32> {
  %0 = vector.load %source[%offset, %offset, %offset]
    : memref<8x16x32xf32>, vector<8xf32>
  return %0 : vector<8xf32>
}

// CHECK-LABEL: @load_1D_vector(
// CHECK-SAME:  %[[SRC:.+]]: memref<8x16x32xf32>,
// CHECK-SAME:  %[[OFFSET:.+]]: index
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], %[[OFFSET]], 0]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[COLLAPSED]]
// CHECK-SAME:    memref<32xf32, strided<[1], offset: ?>> -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]]]{{.*}}-> vector<8xf32>
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
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], 0, 0]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc
// CHECK-SAME:    %[[COLLAPSED]]
// CHECK-SAME:    memref<16x32xf32, strided<[32, 1], offset: ?>> -> !xegpu.tensor_desc<8x16xf32>
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]], %[[OFFSET]]]{{.*}}-> vector<8x16xf32>
// CHECK:       return %[[VEC]]

// -----

func.func @load_dynamic_source(%source: memref<?x?x?xf32>,
    %i: index, %j: index, %k: index) -> vector<8x16xf32> {
  %0 = vector.load %source[%i, %j, %k]
    : memref<?x?x?xf32>, vector<8x16xf32>
  return %0 : vector<8x16xf32>
}

// CHECK-LABEL: @load_dynamic_source(
// CHECK-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0]
// CHECK:       {{.*}} %[[SIZES:.+]]:2, %[[STRIDES:.+]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[COLLAPSED]]
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFF1]], %[[OFF2]]]{{.*}}-> vector<8x16xf32>
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
// CHECK-SAME:    %[[SRC]]
// CHECK-SAME:    memref<7x15xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK:       %[[VEC:.+]] = xegpu.load_nd %[[DESC]][%[[OFFSET]], %[[OFFSET]]]{{.*}}-> vector<8x16xf32>
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
