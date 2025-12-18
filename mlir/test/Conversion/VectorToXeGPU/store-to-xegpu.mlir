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
// CHECK:       %[[ELEM_BYTES:.*]] = arith.constant 4 : index
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], %[[OFFSET]], 0]
// CHECK:       %[[BASE_BUFFER:.+]], %[[OFFSET1:.+]], %[[SIZES:.+]], %[[STRIDES:.+]] = memref.extract_strided_metadata %[[COLLAPSED]]
// CHECK-SAME:    : memref<32xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
// CHECK:       %[[INTPTR:.+]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// CHECK-SAME:    : memref<f32> -> index
// CHECK:       %[[MUL:.+]] = arith.muli %[[OFFSET1]], %[[ELEM_BYTES]] : index
// CHECK:       %[[ADD:.+]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// CHECK:       %[[I64PTR:.+]] = arith.index_cast %[[ADD]] : index to i64
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [32],
// CHECK-SAME:                   strides : [1] : i64  -> !xegpu.tensor_desc<8xf32,
// CHECK-SAME:    boundary_check = false
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFFSET]]] : vector<8xf32>

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
// CHECK:       %[[ELEM_BYTES:.*]] = arith.constant 4 : index
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFFSET]], 0, 0]
// CHECK:       %[[BASE_BUFFER:.*]], %[[OFF1:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// CHECK:       %[[INTPTR:.*]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]]
// CHECK-SAME:    : memref<f32> -> index
// CHECK:       %[[MUL:.+]] = arith.muli %[[OFF1]], %[[ELEM_BYTES]] : index
// CHECK:       %[[ADD:.+]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// CHECK:       %[[I64PTR:.+]] = arith.index_cast %[[ADD]] : index to i64
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [16, 32],
// CHECK-SAME:                   strides : [32, 1] : i64 -> !xegpu.tensor_desc<8x16xf32>
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : vector<8x16xf32>

// -----

func.func @store_dynamic_source(%vec: vector<8x16xf32>,
    %source: memref<?x?x?xf32>, %i: index, %j: index, %k: index) {
  vector.store %vec, %source[%i, %j, %k]
    : memref<?x?x?xf32>, vector<8x16xf32>
  return
}

// CHECK-LABEL: @store_dynamic_source(
// CHECK-SAME:  %[[VEC:.+]]: vector<8x16xf32>,
// CHECK-SAME:  %[[SRC:.+]]: memref<?x?x?xf32>,
// CHECK-SAME:  %[[OFF0:.+]]: index, %[[OFF1:.+]]: index, %[[OFF2:.+]]: index
// CHECK:       %[[ELEM_BYTES:.*]] = arith.constant 4 : index
// CHECK:       %[[COLLAPSED:.+]] = memref.subview %[[SRC]][%[[OFF0]], 0, 0]
// CHECK:       %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.+]]:2, %[[STRIDES:.+]]:2 = memref.extract_strided_metadata %[[COLLAPSED]]
// CHECK:       %[[INTPTR:.+]] = memref.extract_aligned_pointer_as_index %[[BASE_BUFFER]] : memref<f32> -> index
// CHECK:       %[[MUL:.+]] = arith.muli %[[OFFSET]], %[[ELEM_BYTES]] : index
// CHECK:       %[[ADD:.+]] = arith.addi %[[INTPTR]], %[[MUL]] : index
// CHECK:       %[[I64PTR:.+]] = arith.index_cast %[[ADD]] : index to i64
// CHECK:       %[[DESC:.+]] = xegpu.create_nd_tdesc %[[I64PTR]], shape : [%[[SIZES]]#0, %[[SIZES]]#1],
// CHECK-SAME:                   strides : [%[[STRIDES]]#0, 1] : i64 -> !xegpu.tensor
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFF1]], %[[OFF2]]] : vector<8x16xf32>

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
// CHECK-SAME:    %[[SRC]]
// CHECK-SAME:    memref<7x64xf32> -> !xegpu.tensor_desc<8x16xf32>
// CHECK:       xegpu.store_nd %[[VEC]], %[[DESC]][%[[OFFSET]], %[[OFFSET]]] : vector<8x16xf32>

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
