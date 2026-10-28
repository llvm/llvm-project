// RUN: mlir-opt --test-xegpu-array-length-optimization --split-input-file %s | FileCheck %s

gpu.module @test {
// CHECK-LABEL: func.func @test_load_nd_with_extract_slice
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf16>)
func.func @test_load_nd_with_extract_slice(%arg0: memref<4096x4096xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]]
  // CHECK-SAME: memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>

  // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<64x16xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>

  // Extract first 16x16 block (memory layout: [0:16][0:16])
  // In memory layout this is first half of FCD
  // In register layout this stays [0:16][0:16]
  // CHECK: %[[EXTRACT0:.*]] = vector.extract_strided_slice %[[LOAD]]
  // CHECK-SAME: offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
  %extract0 = vector.extract_strided_slice %load offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  return %extract0 : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_load_nd_with_second_extract
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf16>)
func.func @test_load_nd_with_second_extract(%arg0: memref<4096x4096xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]]
  // CHECK-SAME: memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>

  // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>> -> vector<64x16xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>

  // Extract second 16x16 block (memory layout: [0:16][16:32])
  // In memory layout this is second half of FCD
  // In register layout this should be [32:48][0:16] (second array element)
  // array_index = 16 / 16 = 1
  // new_offset0 = 0 + (1 * 32) = 32
  // new_offset1 = 16 % 16 = 0
  // CHECK: %[[EXTRACT1:.*]] = vector.extract_strided_slice %[[LOAD]]
  // CHECK-SAME: offsets = [32, 0], sizes = [16, 16], strides = [1, 1]
  %extract1 = vector.extract_strided_slice %load offsets = [0, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  return %extract1 : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_prefetch_nd_32x32
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf16>)
func.func @test_prefetch_nd_32x32(%arg0: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]]
  // CHECK-SAME: memref<4096x4096xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>

  // CHECK: xegpu.prefetch_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  xegpu.prefetch_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16>

  return
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_no_optimization_16x16
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf16>)
func.func @test_no_optimization_16x16(%arg0: memref<4096x4096xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]]
  // CHECK-SAME: memref<4096x4096xf16> -> !xegpu.tensor_desc<16x16xf16>
  // CHECK-NOT: array_length
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<16x16xf16>

  // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<16x16xf16> -> vector<16x16xf16>

  return %load : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// Loads that carry a non-identity transpose must not be rewritten: the array
// blocks would otherwise be stacked along the non-FCD dimension, which
// conflicts with the transpose semantics.
// CHECK-LABEL: func.func @test_no_optimization_with_transpose
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf32>)
func.func @test_no_optimization_with_transpose(%arg0: memref<4096x4096xf32>) -> vector<32x32xf32> {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]]
  // CHECK-SAME: memref<4096x4096xf32> -> !xegpu.tensor_desc<32x32xf32>
  // CHECK-NOT: array_length
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<32x32xf32>

  // CHECK: xegpu.load_nd %[[TDESC]]
  // CHECK-SAME: <{transpose = array<i64: 1, 0>}>
  // CHECK-SAME: !xegpu.tensor_desc<32x32xf32> -> vector<32x32xf32>
  %load = xegpu.load_nd %tdesc[%c0, %c0] <{transpose = array<i64: 1, 0>}> : !xegpu.tensor_desc<32x32xf32> -> vector<32x32xf32>

  return %load : vector<32x32xf32>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_multiple_extracts
// CHECK-SAME:    (%[[ARG0:.*]]: memref<4096x4096xf16>)
func.func @test_multiple_extracts(%arg0: memref<4096x4096xf16>) -> (vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16>) {
  %c0 = arith.constant 0 : index

  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>

  // Memory layout view (32x32):
  //   [0:16][0:16]   | [0:16][16:32]
  //   [16:32][0:16]  | [16:32][16:32]
  //
  // Register layout view (64x16):
  //   [0:16][0:16]    (first array element, first half)
  //   [16:32][0:16]   (first array element, second half)
  //   [32:48][0:16]   (second array element, first half)
  //   [48:64][0:16]   (second array element, second half)

  // Extract [0:16][0:16] -> register [0:16][0:16]
  // CHECK: vector.extract_strided_slice
  // CHECK-SAME: offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
  %e0 = vector.extract_strided_slice %load offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  // Extract [0:16][16:32] -> register [32:48][0:16]
  // CHECK: vector.extract_strided_slice
  // CHECK-SAME: offsets = [32, 0], sizes = [16, 16], strides = [1, 1]
  %e1 = vector.extract_strided_slice %load offsets = [0, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  // Extract [16:32][0:16] -> register [16:32][0:16]
  // CHECK: vector.extract_strided_slice
  // CHECK-SAME: offsets = [16, 0], sizes = [16, 16], strides = [1, 1]
  %e2 = vector.extract_strided_slice %load offsets = [16, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  // Extract [16:32][16:32] -> register [48:64][0:16]
  // CHECK: vector.extract_strided_slice
  // CHECK-SAME: offsets = [48, 0], sizes = [16, 16], strides = [1, 1]
  %e3 = vector.extract_strided_slice %load offsets = [16, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  return %e0, %e1, %e2, %e3 : vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16>, vector<16x16xf16>
}
}

// -----

gpu.module @test {
// Pointer-form (i64) source — shape/strides are given explicitly and must be
// carried through to the rewritten create_nd_tdesc.
// CHECK-LABEL: func.func @test_pointer_source
// CHECK-SAME:    (%[[ARG0:.*]]: i64)
func.func @test_pointer_source(%arg0: i64) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index

  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]], shape : [64, 64], strides : [64, 1]
  // CHECK-SAME: i64 -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64, boundary_check = false>>
  %tdesc = xegpu.create_nd_tdesc %arg0, shape : [64, 64], strides : [64, 1] : i64 -> !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>

  // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: -> vector<64x16xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x32xf16>

  // CHECK: vector.extract_strided_slice %[[LOAD]]
  // CHECK-SAME: offsets = [32, 0], sizes = [16, 16], strides = [1, 1]
  %e = vector.extract_strided_slice %load offsets = [0, 16], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>

  return %e : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// Dynamic-shape memref source — shape/strides are given via operands and must
// be carried through.
// CHECK-LABEL: func.func @test_dynamic_memref_source
// CHECK-SAME:    (%[[ARG0:.*]]: memref<?x?xf16>, %[[H:.*]]: index, %[[W:.*]]: index)
func.func @test_dynamic_memref_source(%arg0: memref<?x?xf16>, %h: index, %w: index) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index

  // A dynamic memref uses the bare form; its shape/strides come from the memref.
  // CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc %[[ARG0]] : memref<?x?xf16> -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
  %tdesc = xegpu.create_nd_tdesc %arg0 : memref<?x?xf16> -> !xegpu.tensor_desc<32x32xf16>

  // CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]][%{{.*}}, %{{.*}}]
  // CHECK-SAME: -> vector<64x16xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>

  %e = vector.extract_strided_slice %load offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : vector<32x32xf16> to vector<16x16xf16>
  return %e : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_multiple_loads
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 2 : i64>>
// CHECK: %[[LOAD0:.*]] = xegpu.load_nd %[[TDESC]]
// CHECK-SAME: -> vector<64x16xf16>
// CHECK: vector.extract_strided_slice %[[LOAD0]]
// CHECK-SAME: offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
// CHECK: %[[LOAD1:.*]] = xegpu.load_nd %[[TDESC]]
// CHECK-SAME: -> vector<64x16xf16>
// CHECK: vector.extract_strided_slice %[[LOAD1]]
// CHECK-SAME: offsets = [32, 0], sizes = [16, 16], strides = [1, 1]
func.func @test_multiple_loads(%arg0: memref<4096x4096xf16>)
    -> (vector<16x16xf16>, vector<16x16xf16>) {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>
  %load0 = xegpu.load_nd %tdesc[%c0, %c0]
      : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
  %e0 = vector.extract_strided_slice %load0
      offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : vector<32x32xf16> to vector<16x16xf16>
  %load1 = xegpu.load_nd %tdesc[%c0, %c0]
      : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
  %e1 = vector.extract_strided_slice %load1
      offsets = [0, 16], sizes = [16, 16], strides = [1, 1]
      : vector<32x32xf16> to vector<16x16xf16>
  return %e0, %e1 : vector<16x16xf16>, vector<16x16xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_cross_block_extract
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x32xf16>
// CHECK-NOT: array_length
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
// CHECK-SAME: -> vector<32x32xf16>
// CHECK: vector.extract_strided_slice %[[LOAD]]
// CHECK-SAME: offsets = [0, 8], sizes = [16, 16], strides = [1, 1]
func.func @test_cross_block_extract(%arg0: memref<4096x4096xf16>)
    -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0]
      : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
  %e = vector.extract_strided_slice %load
      offsets = [0, 8], sizes = [16, 16], strides = [1, 1]
      : vector<32x32xf16> to vector<16x16xf16>
  return %e : vector<16x16xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_unsupported_load_user
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x32xf16>
// CHECK-NOT: array_length
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
// CHECK-SAME: -> vector<32x32xf16>
// CHECK: vector.shape_cast %[[LOAD]] : vector<32x32xf16> to vector<1024xf16>
func.func @test_unsupported_load_user(%arg0: memref<4096x4096xf16>)
    -> vector<1024xf16> {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0]
      : !xegpu.tensor_desc<32x32xf16> -> vector<32x32xf16>
  %cast = vector.shape_cast %load
      : vector<32x32xf16> to vector<1024xf16>
  return %cast : vector<1024xf16>
}
}

// -----

gpu.module @test {
// CHECK-LABEL: func.func @test_unsupported_descriptor_user
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x32xf16>
// CHECK-NOT: array_length
// CHECK: "test.use"(%[[TDESC]])
func.func @test_unsupported_descriptor_user(%arg0: memref<4096x4096xf16>) {
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x32xf16>
  "test.use"(%tdesc) : (!xegpu.tensor_desc<32x32xf16>) -> ()
  return
}
}

// -----

#layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>

gpu.module @test {
// CHECK-LABEL: func.func @test_incompatible_descriptor_layout
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x64xf16, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>>
// CHECK-NOT: array_length
func.func @test_incompatible_descriptor_layout(
    %arg0: memref<4096x4096xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x64xf16, #layout>
  %load = xegpu.load_nd %tdesc[%c0, %c0]
      : !xegpu.tensor_desc<32x64xf16, #layout> -> vector<32x64xf16>
  %e = vector.extract_strided_slice %load
      offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : vector<32x64xf16> to vector<16x16xf16>
  return %e : vector<16x16xf16>
}
}

// -----

#layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>

gpu.module @test {
// CHECK-LABEL: func.func @test_incompatible_load_layout
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x64xf16>
// CHECK-NOT: array_length
// CHECK: xegpu.load_nd %[[TDESC]]
// CHECK-SAME: <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>}>
func.func @test_incompatible_load_layout(%arg0: memref<4096x4096xf16>)
    -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x64xf16>
  %load = xegpu.load_nd %tdesc[%c0, %c0] <{layout = #layout}>
      : !xegpu.tensor_desc<32x64xf16> -> vector<32x64xf16>
  %e = vector.extract_strided_slice %load
      offsets = [0, 0], sizes = [16, 16], strides = [1, 1]
      : vector<32x64xf16> to vector<16x16xf16>
  return %e : vector<16x16xf16>
}
}

// -----

#layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>

gpu.module @test {
// CHECK-LABEL: func.func @test_incompatible_prefetch_layout
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x64xf16>
// CHECK-NOT: array_length
// CHECK: xegpu.prefetch_nd %[[TDESC]]
// CHECK-SAME: <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 2]>}>
func.func @test_incompatible_prefetch_layout(%arg0: memref<4096x4096xf16>) {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16> -> !xegpu.tensor_desc<32x64xf16>
  xegpu.prefetch_nd %tdesc[%c0, %c0] <{layout = #layout}>
      : !xegpu.tensor_desc<32x64xf16>
  return
}
}

// -----

#layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>

gpu.module @test {
// CHECK-LABEL: func.func @test_compatible_layouts
// CHECK: %[[TDESC:.*]] = xegpu.create_nd_tdesc
// CHECK-SAME: -> !xegpu.tensor_desc<32x16xf16, #xegpu.block_tdesc_attr<array_length = 4 : i64>, #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>>
// CHECK: xegpu.prefetch_nd %[[TDESC]]
// CHECK-SAME: <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}>
// CHECK: %[[LOAD:.*]] = xegpu.load_nd %[[TDESC]]
// CHECK-SAME: <{layout = #xegpu.layout<lane_layout = [1, 16], lane_data = [1, 1]>}>
// CHECK-SAME: -> vector<128x16xf16>
// CHECK: vector.extract_strided_slice %[[LOAD]]
// CHECK-SAME: offsets = [96, 0], sizes = [16, 16], strides = [1, 1]
func.func @test_compatible_layouts(%arg0: memref<4096x4096xf16>)
    -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %tdesc = xegpu.create_nd_tdesc %arg0
      : memref<4096x4096xf16>
        -> !xegpu.tensor_desc<32x64xf16, #layout>
  xegpu.prefetch_nd %tdesc[%c0, %c0] <{layout = #layout}>
      : !xegpu.tensor_desc<32x64xf16, #layout>
  %load = xegpu.load_nd %tdesc[%c0, %c0] <{layout = #layout}>
      : !xegpu.tensor_desc<32x64xf16, #layout> -> vector<32x64xf16>
  %e = vector.extract_strided_slice %load
      offsets = [0, 48], sizes = [16, 16], strides = [1, 1]
      : vector<32x64xf16> to vector<16x16xf16>
  return %e : vector<16x16xf16>
}
}
