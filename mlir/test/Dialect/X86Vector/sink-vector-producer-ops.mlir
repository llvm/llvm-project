// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

func.func @sink_vector_loads(%arg0: memref<16x16xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = vector.load %arg0[%c0, %c0] : memref<16x16xf32>, vector<8xf32>
  %1 = vector.load %arg0[%c0, %c8] : memref<16x16xf32>, vector<8xf32>
  %2 = vector.load %arg0[%c8, %c0] : memref<16x16xf32>, vector<8xf32>
  %3 = vector.load %arg0[%c8, %c8] : memref<16x16xf32>, vector<8xf32>
  %4 = vector.fma %0, %1, %arg1 : vector<8xf32>
  %5 = vector.fma %2, %3, %4 : vector<8xf32>
  return %5 : vector<8xf32>
}

// CHECK-LABEL: @sink_vector_loads
// CHECK: vector.load
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

func.func @sink_vector_transfer_reads(%arg0: memref<16x16xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true]} : memref<16x16xf32>, vector<8xf32>
  %2 = vector.transfer_read %arg0[%c0, %c8], %0 {in_bounds = [true]} : memref<16x16xf32>, vector<8xf32>
  %3 = vector.transfer_read %arg0[%c8, %c0], %0 {in_bounds = [true]} : memref<16x16xf32>, vector<8xf32>
  %4 = vector.transfer_read %arg0[%c8, %c8], %0 {in_bounds = [true]} : memref<16x16xf32>, vector<8xf32>
  %5 = vector.fma %1, %2, %arg1 : vector<8xf32>
  %6 = vector.fma %3, %4, %5 : vector<8xf32>
  return %6 : vector<8xf32>
}

// CHECK-LABEL: @sink_vector_transfer_reads
// CHECK: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

func.func @sink_vector_transfer_reads_tensor(%arg0: tensor<16x16xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true]} : tensor<16x16xf32>, vector<8xf32>
  %2 = vector.transfer_read %arg0[%c0, %c8], %0 {in_bounds = [true]} : tensor<16x16xf32>, vector<8xf32>
  %3 = vector.transfer_read %arg0[%c8, %c0], %0 {in_bounds = [true]} : tensor<16x16xf32>, vector<8xf32>
  %4 = vector.transfer_read %arg0[%c8, %c8], %0 {in_bounds = [true]} : tensor<16x16xf32>, vector<8xf32>
  %5 = vector.fma %1, %2, %arg1 : vector<8xf32>
  %6 = vector.fma %3, %4, %5 : vector<8xf32>
  return %6 : vector<8xf32>
}

// CHECK-LABEL: @sink_vector_transfer_reads_tensor
// CHECK: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.fma
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

func.func @sink_vector_transfer_reads_bf16(%arg0: tensor<4x64x32x2xbf16>, %arg1: tensor<4x32x64x2xbf16>, %arg2: vector<1x16xf32>) -> vector<1x16xf32> {
  %0 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %extracted_slice = tensor.extract_slice %arg0[%c0, %c0, %c0, 0] [1, 4, 1, 2] [1, 1, 1, 1] : tensor<4x64x32x2xbf16> to tensor<1x4x1x2xbf16>
  %extracted_slice_0 = tensor.extract_slice %arg1[%c0, %c0, %c0, 0] [1, 1, 32, 2] [1, 1, 1, 1] : tensor<4x32x64x2xbf16> to tensor<1x1x32x2xbf16>
  %1 = vector.transfer_read %extracted_slice[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<1x4x1x2xbf16>, vector<1x1x1x2xbf16>
  %2 = vector.transfer_read %extracted_slice[%c0, %c1, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<1x4x1x2xbf16>, vector<1x1x1x2xbf16>
  %3 = vector.transfer_read %extracted_slice_0[%c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<1x1x32x2xbf16>, vector<1x1x16x2xbf16>
  %4 = vector.transfer_read %extracted_slice_0[%c0, %c0, %c16, %c0], %0 {in_bounds = [true, true, true, true]} : tensor<1x1x32x2xbf16>, vector<1x1x16x2xbf16>
  %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %3, %arg2 {unroll_shape = array<i64: 1, 2, 1, 16, 1>} : vector<1x1x1x2xbf16>, vector<1x1x16x2xbf16> into vector<1x16xf32>
  %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %4, %5 {unroll_shape = array<i64: 1, 2, 1, 16, 1>} : vector<1x1x1x2xbf16>, vector<1x1x16x2xbf16> into vector<1x16xf32>
  %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %3, %6 {unroll_shape = array<i64: 1, 2, 1, 16, 1>} : vector<1x1x1x2xbf16>, vector<1x1x16x2xbf16> into vector<1x16xf32>
  %8 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %2, %4, %7 {unroll_shape = array<i64: 1, 2, 1, 16, 1>} : vector<1x1x1x2xbf16>, vector<1x1x16x2xbf16> into vector<1x16xf32>
  return %8 : vector<1x16xf32>
}

// CHECK-LABEL: @sink_vector_transfer_reads_bf16
// CHECK: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.contract
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.contract
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.contract
// CHECK-NEXT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

func.func @negative_no_infinite_looping(%arg0: memref<16x16xf32>, %arg1: vector<8xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = vector.load %arg0[%c0, %c0] : memref<16x16xf32>, vector<8xf32>
  %1 = vector.load %arg0[%c0, %c8] : memref<16x16xf32>, vector<8xf32>
  %2 = vector.fma %0, %1, %arg1 : vector<8xf32>
  return %2: vector<8xf32>
}

// CHECK-LABEL: @negative_no_infinite_looping
// CHECK: vector.load
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

// -----

func.func @negative_no_sink_outside_block(%arg0: memref<8x16xf32>, %arg1: i1) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = vector.load %arg0[%c0, %c0] : memref<8x16xf32>, vector<8xf32>
  %1 = vector.load %arg0[%c0, %c8] : memref<8x16xf32>, vector<8xf32>
  %2 = scf.if %arg1 -> (vector<8xf32>) {
    scf.yield %0 : vector<8xf32>
  } else {
    scf.yield %1 : vector<8xf32>
  }
  return %2 : vector<8xf32>
}

// CHECK-LABEL: @negative_no_sink_outside_block
// CHECK: vector.load
// CHECK-NEXT: vector.load
// CHECK-NEXT: scf.if
// CHECK-NEXT: scf.yield

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.x86vector.sink_vector_producer_ops
    } : !transform.any_op
    transform.yield
  }
}

