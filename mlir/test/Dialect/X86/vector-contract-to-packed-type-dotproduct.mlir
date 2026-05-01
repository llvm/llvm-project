// RUN: mlir-opt %s -transform-interpreter -cse -split-input-file | FileCheck %s

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<1x16xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_bf16dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @brgemm_to_bf16dp
// CHECK: vector.broadcast
// CHECK: x86.avx512.dot

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x1x2xbf16>
!vecB = vector<1x1x1x2xbf16>
!vecC = vector<16x1xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_bf16dp_bcst_B(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @brgemm_to_bf16dp_bcst_B
// CHECK: vector.broadcast
// CHECK: x86.avx512.dot

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xi8>
!vecB = vector<1x1x16x4xi8>
!vecC = vector<1x16xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_avx10int8dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @brgemm_to_avx10int8dp
// CHECK: vector.broadcast
// CHECK: x86.avx10.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x16x1x4xi8>
!vecB = vector<1x1x1x4xi8>
!vecC = vector<1x16x1xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_avx10int8dp_bcst_B(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}


// CHECK-LABEL: @batch_matmul_avx10int8dp_bcst_B
// CHECK: vector.broadcast
// CHECK: x86.avx10.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xi8>
!vecB = vector<1x1x8x4xi8>
!vecC = vector<1x8xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_int8dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @brgemm_to_int8dp
// CHECK: vector.broadcast
// CHECK: x86.avx.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<1x1x16xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_bf16dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @batch_matmul_bf16dp
// CHECK: vector.broadcast
// CHECK: x86.avx512.dot

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xi8>
!vecB = vector<1x1x8x4xi8>
!vecC = vector<1x1x8xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_int8dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}


// CHECK-LABEL: @batch_matmul_int8dp
// CHECK: vector.broadcast
// CHECK: x86.avx.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x8x1x4xi8>
!vecB = vector<1x1x1x4xi8>
!vecC = vector<1x8x1xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_int8dp_bcst_B(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}


// CHECK-LABEL: @batch_matmul_int8dp_bcst_B
// CHECK: vector.broadcast
// CHECK: x86.avx.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x16x2xbf16>
!vecC = vector<1x16xf32>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_outer_product_to_bf16dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @matmul_outer_product_to_bf16dp
// CHECK: vector.broadcast
// CHECK: x86.avx512.dot

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x4xi8>
!vecB = vector<1x8x4xi8>
!vecC = vector<1x8xi32>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @matmul_outer_product_to_int8dp(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @matmul_outer_product_to_int8dp
// CHECK: vector.broadcast
// CHECK: x86.avx.dot.i8

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_bf16dp_flat_layout(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %3 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %4 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %5 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %4, %2
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %5, %3
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_bf16dp_flat_layout
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x4xi8>
!vecB = vector<4x16xi8>
!vecC = vector<1x16xi32>
!memrefA = memref<4x4xi8>
!memrefB = memref<4x64xi8>
!memrefC = memref<4x64xi32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_i8_avx10dp_flat_layout(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %3 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %4 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %5 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %4, %2
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %5, %3
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_i8_avx10dp_flat_layout
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>
// CHECK: vector.shuffle{{.*}}[0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, 3, 35, 67, 99, 8, 40, 72, 104, 9, 41, 73, 105, 10, 42, 74, 106, 11, 43, 75, 107, 16, 48, 80, 112, 17, 49, 81, 113, 18, 50, 82, 114, 19, 51, 83, 115, 24, 56, 88, 120, 25, 57, 89, 121, 26, 58, 90, 122, 27, 59, 91, 123] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 68, 100, 5, 37, 69, 101, 6, 38, 70, 102, 7, 39, 71, 103, 12, 44, 76, 108, 13, 45, 77, 109, 14, 46, 78, 110, 15, 47, 79, 111, 20, 52, 84, 116, 21, 53, 85, 117, 22, 54, 86, 118, 23, 55, 87, 119, 28, 60, 92, 124, 29, 61, 93, 125, 30, 62, 94, 126, 31, 63, 95, 127] : vector<64xi8>, vector<64xi8>
// CHECK: x86.avx10.dot.i8
// CHECK: x86.avx10.dot.i8
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x4xi8>
!vecB = vector<4x8xi8>
!vecC = vector<1x8xi32>
!memrefA = memref<4x4xi8>
!memrefB = memref<4x64xi8>
!memrefC = memref<4x64xi32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_i8_avx2dp_flat_layout(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %3 = vector.transfer_read %arg2[%c0, %c8], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %4 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %5 = vector.transfer_read %arg1[%c0, %c8], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %4, %2
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %5, %3
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c8] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_i8_avx2dp_flat_layout
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 8, 9, 10, 11] : vector<8xi32>, vector<8xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 5, 6, 7, 12, 13, 14, 15] : vector<8xi32>, vector<8xi32>
// CHECK: vector.shuffle{{.*}}[0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59] : vector<32xi8>, vector<32xi8>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63] : vector<32xi8>, vector<32xi8>
// CHECK: x86.avx.dot.i8
// CHECK: x86.avx.dot.i8
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 8, 9, 10, 11] : vector<8xi32>, vector<8xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 5, 6, 7, 12, 13, 14, 15] : vector<8xi32>, vector<8xi32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_bf16dp_flat_layout_offset_args_and_read_after_vc(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC, %arg3: index) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %3 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %4 = vector.transfer_read %arg1[%arg3, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %5 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %4, %2
    : !vecA, !vecB into !vecC

  %6 = vector.transfer_read %arg1[%arg3, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %6, %3
    : !vecA, !vecB into !vecC

  vector.transfer_write %5, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_bf16dp_flat_layout_offset_args_and_read_after_vc
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<1x1x2xbf16, strided<[2048, 32, 1], offset: ?>>
!memrefB = memref<1x2x32xbf16, strided<[2048, 64, 1], offset: ?>>
!memrefC = memref<1x32xf32, strided<[64, 1], offset: ?>>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @brmatmul_bf16dp_flat_layout_loop(%arg0: memref<16x64x32xbf16>, %arg1: memref<16x32x64xbf16>,
                             %arg2: memref<64x64xf32>) -> memref<64x64xf32> {
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %arg3 = %c0 to %c64 step %c1 {
    scf.for %arg4 = %c0 to %c64 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [1, 32] [1, 1] 
			: memref<64x64xf32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} 
			: !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]} 
			: !memrefC, !vecC

      %4:2 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3) -> (!vecC, !vecC) {
        %5:2 = scf.for %arg8 = %c0 to %c32 step %c2 iter_args(%arg9 = %arg6, %arg10 = %arg7) -> (!vecC, !vecC) {

          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg8] [1, 1, 2] [1, 1, 1] 
			: memref<16x64x32xbf16> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg8, %arg4] [1, 2, 32] [1, 1, 1] 
			: memref<16x32x64xbf16> to !memrefB

          %6 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} 
			: !memrefA, !vecA
          %7 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} 
			: !memrefB, !vecB
          %8 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1 {in_bounds = [true, true, true]} 
			: !memrefB, !vecB

          %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = 
			["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %7, %arg9 
			{unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          %10 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = 
			["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %6, %8, %arg10 
			{unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC

          scf.yield %9, %10 : !vecC, !vecC
        }
        scf.yield %5#0, %5#1 : !vecC, !vecC
      }

      vector.transfer_write %4#1, %subview[%c0, %c16] {in_bounds = [true, true]} 
			: !vecC, !memrefC
      vector.transfer_write %4#0, %subview[%c0, %c0] {in_bounds = [true, true]} 
			: !vecC, !memrefC
    }
  }

  return %arg2 : memref<64x64xf32>
}

// CHECK-LABEL: @brmatmul_bf16dp_flat_layout_loop
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @matmul_bf16dp_flat_layout_B_shuffled(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c16] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c16] :
        !memrefC, !vecC

  %6 = vector.shape_cast %2 : !vecB to vector<32xbf16>
  %7 = vector.shape_cast %3 : !vecB to vector<32xbf16>
  %8 = vector.shuffle %6, %7 [0, 32, 1, 33, 2, 34, 3, 35,
        8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18,
        50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] :
        vector<32xbf16>, vector<32xbf16>
  %9 = vector.shuffle %6, %7 [4, 36, 5, 37, 6, 38, 7, 39,
        12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53,
        22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] :
        vector<32xbf16>, vector<32xbf16>

  %10 = vector.shape_cast %8 : vector<32xbf16> to !vecB
  %11 = vector.shape_cast %9 : vector<32xbf16> to !vecB

  %12 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %10, %4
    : !vecA, !vecB into !vecC

  %13 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %11, %5
    : !vecA, !vecB into !vecC

  vector.store %12, %arg2[%c0, %c0]  : !memrefC, !vecC
  vector.store %13, %arg2[%c0, %c16]  : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @matmul_bf16dp_flat_layout_B_shuffled
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK-NOT: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x4xi8>
!vecB = vector<1x4x16xi8>
!vecC = vector<1x16xi32>
!memrefA = memref<1x2x4xi8, strided<[16384, 256, 1], offset: ?>>
!memrefB = memref<1x4x32xi8, strided<[32768, 128, 1], offset: ?>>
!memrefC = memref<2x32xi32, strided<[128, 1], offset: ?>>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @brgemm_int8_flat_avx10(%arg0: memref<16x64x256xi8>, %arg1: memref<16x256x128xi8>, %arg2: memref<64x128xi32>) -> memref<64x128xi32> {
  %0 = ub.poison : i32
  %1 = ub.poison : i8
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c16 = arith.constant 16 : index
  %c256 = arith.constant 256 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %arg3 = %c0 to %c64 step %c2 {
    scf.for %arg4 = %c0 to %c128 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [2, 32] [1, 1] : memref<64x128xi32> to !memrefC
      %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %3 = vector.transfer_read %subview[%c0, %c16], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %4 = vector.transfer_read %subview[%c1, %c0], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %5 = vector.transfer_read %subview[%c1, %c16], %0 {in_bounds = [true, true]}
                : !memrefC, !vecC
      %6:4 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5) -> (!vecC, !vecC, !vecC, !vecC) {
        %7:4 = scf.for %arg10 = %c0 to %c256 step %c4 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (!vecC, !vecC, !vecC, !vecC) {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg10] [1, 2, 4] [1, 1, 1] : memref<16x64x256xi8> to !memrefA
          %subview_1 = memref.subview %arg1[%arg5, %arg10, %arg4] [1, 4, 32] [1, 1, 1] : memref<16x256x128xi8> to !memrefB
          %8 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                : !memrefA, !vecA
          %9 = vector.transfer_read %subview_0[%c0, %c1, %c0], %1 {in_bounds = [true, true, true]}
                : !memrefA, !vecA
          %10 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                : !memrefB, !vecB
          %11 = vector.transfer_read %subview_1[%c0, %c0, %c16], %1 {in_bounds = [true, true, true]}
                : !memrefB, !vecB
          %12 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %10, %arg11 {unroll_shape = array<i64: 1, 1, 16, 4>} : !vecA, !vecB into !vecC
          %13 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %8, %11, %arg12 {unroll_shape = array<i64: 1, 1, 16, 4>} : !vecA, !vecB into !vecC
          %14 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %9, %10, %arg13 {unroll_shape = array<i64: 1, 1, 16, 4>} : !vecA, !vecB into !vecC
          %15 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>}
                %9, %11, %arg14 {unroll_shape = array<i64: 1, 1, 16, 4>} : !vecA, !vecB into !vecC
          scf.yield %12, %13, %14, %15 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %7#0, %7#1, %7#2, %7#3 : !vecC, !vecC, !vecC, !vecC
      }
      vector.transfer_write %6#3, %subview[%c1, %c16] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#2, %subview[%c1, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#1, %subview[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC
      vector.transfer_write %6#0, %subview[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
    }
  }
  %alloc = memref.alloc() : memref<64x128xi32>
  memref.copy %arg2, %alloc : memref<64x128xi32> to memref<64x128xi32>
  return %alloc : memref<64x128xi32>
}

// CHECK-LABEL: @brgemm_int8_flat_avx10
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>
// CHECK: vector.shuffle{{.*}}[0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, 3, 35, 67, 99, 8, 40, 72, 104, 9, 41, 73, 105, 10, 42, 74, 106, 11, 43, 75, 107, 16, 48, 80, 112, 17, 49, 81, 113, 18, 50, 82, 114, 19, 51, 83, 115, 24, 56, 88, 120, 25, 57, 89, 121, 26, 58, 90, 122, 27, 59, 91, 123] : vector<64xi8>, vector<64xi8>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 68, 100, 5, 37, 69, 101, 6, 38, 70, 102, 7, 39, 71, 103, 12, 44, 76, 108, 13, 45, 77, 109, 14, 46, 78, 110, 15, 47, 79, 111, 20, 52, 84, 116, 21, 53, 85, 117, 22, 54, 86, 118, 23, 55, 87, 119, 28, 60, 92, 124, 29, 61, 93, 125, 30, 62, 94, 126, 31, 63, 95, 127] : vector<64xi8>, vector<64xi8>
// CHECK: x86.avx10.dot.i8
// CHECK: x86.avx10.dot.i8
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xi32>, vector<16xi32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xi32>, vector<16xi32>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x2x16xbf16>
!vecC = vector<1x16xf32>

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @brmatmul_bf16dp_flat_layout_loop_tensor_type(%arg0: tensor<16x64x32xbf16>, %arg1: tensor<16x32x64xbf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = ub.poison : f32
  %1 = ub.poison : bf16
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %2 = scf.for %arg3 = %c0 to %c64 step %c2 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
    %3 = scf.for %arg5 = %c0 to %c64 step %c32 iter_args(%arg6 = %arg4) -> (tensor<64x64xf32>) {

      %extracted_slice = tensor.extract_slice %arg6[%arg3, %arg5] [2, 32] [1, 1]
                                : tensor<64x64xf32> to tensor<2x32xf32>
      %4 = vector.transfer_read %extracted_slice[%c0, %c0], %0 {in_bounds = [true, true]}
                                : tensor<2x32xf32>, !vecC
      %5 = vector.transfer_read %extracted_slice[%c0, %c16], %0 {in_bounds = [true, true]}
                                : tensor<2x32xf32>, !vecC
      %6 = vector.transfer_read %extracted_slice[%c1, %c0], %0 {in_bounds = [true, true]}
                                : tensor<2x32xf32>, !vecC
      %7 = vector.transfer_read %extracted_slice[%c1, %c16], %0 {in_bounds = [true, true]}
                                : tensor<2x32xf32>, !vecC
      %8:4 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %4, %arg9 = %5, %arg10 = %6, %arg11 = %7) -> (!vecC, !vecC, !vecC, !vecC) {
        %13:4 = scf.for %arg12 = %c0 to %c32 step %c2 iter_args(%arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10, %arg16 = %arg11) -> (!vecC, !vecC, !vecC, !vecC) {

          %extracted_slice_0 = tensor.extract_slice %arg0[%arg7, %arg3, %arg12] [1, 2, 2] [1, 1, 1]
                                : tensor<16x64x32xbf16> to tensor<1x2x2xbf16>
          %extracted_slice_1 = tensor.extract_slice %arg1[%arg7, %arg12, %arg5] [1, 2, 32] [1, 1, 1]
                                : tensor<16x32x64xbf16> to tensor<1x2x32xbf16>
          %14 = vector.transfer_read %extracted_slice_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                                : tensor<1x2x2xbf16>, !vecA
          %15 = vector.transfer_read %extracted_slice_0[%c0, %c1, %c0], %1 {in_bounds = [true, true, true]}
                                : tensor<1x2x2xbf16>, !vecA
          %16 = vector.transfer_read %extracted_slice_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]}
                                : tensor<1x2x32xbf16>, !vecB
          %17 = vector.transfer_read %extracted_slice_1[%c0, %c0, %c16], %1 {in_bounds = [true, true, true]}
                                : tensor<1x2x32xbf16>, !vecB
          %18 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %14, %16, %arg13
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          %19 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %14, %17, %arg14
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          %20 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %16, %arg15
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          %21 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types =
                        ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %15, %17, %arg16
                        {unroll_shape = array<i64: 1, 1, 16, 2>} : !vecA, !vecB into !vecC
          scf.yield %18, %19, %20, %21 : !vecC, !vecC, !vecC, !vecC
        }
        scf.yield %13#0, %13#1, %13#2, %13#3 : !vecC, !vecC, !vecC, !vecC
      }

      %9 = vector.transfer_write %8#3, %extracted_slice[%c1, %c16] {in_bounds = [true, true]}
                : !vecC, tensor<2x32xf32>
      %10 = vector.transfer_write %8#2, %9[%c1, %c0] {in_bounds = [true, true]}
                : !vecC, tensor<2x32xf32>
      %11 = vector.transfer_write %8#1, %10[%c0, %c16] {in_bounds = [true, true]}
                : !vecC, tensor<2x32xf32>
      %12 = vector.transfer_write %8#0, %11[%c0, %c0] {in_bounds = [true, true]}
                : !vecC, tensor<2x32xf32>
      %inserted_slice = tensor.insert_slice %12 into %arg6[%arg3, %arg5] [2, 32] [1, 1]
                : tensor<2x32xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    scf.yield %3 : tensor<64x64xf32>
  }
  return %2 : tensor<64x64xf32>
}

// CHECK-LABEL: @brmatmul_bf16dp_flat_layout_loop_tensor_type
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK: x86.avx512.dot
// CHECK: x86.avx512.dot
// CHECK: scf.yield
// CHECK: scf.yield
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK-NEXT: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x2xbf16>
!vecB = vector<1x16x2xbf16>
!vecC = vector<1x16xf32>
#map = affine_map<(d4, d1, d2, d3) -> (d1, d3, d4)>
#map1 = affine_map<(d4, d1, d2, d3) -> (d3, d2, d4)>
#map2 = affine_map<(d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_invalid_vc_kind(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<mul>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_invalid_vc_kind
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xbf16>
!vecB = vector<1x1x16x4xbf16>
!vecC = vector<1x16xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_false_vnni_bf16(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_false_vnni_bf16
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xi8>
!vecB = vector<1x1x8x2xi8>
!vecC = vector<1x8xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_false_vnni_int8(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_false_vnni_int8
// CHECK-NOT: x86.avx.dot.i8
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<3x1x1x2xbf16>
!vecB = vector<3x1x16x2xbf16>
!vecC = vector<3x1x16xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_batch_dimension(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_batch_dimension
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<2x1x1x4xi8>
!vecB = vector<2x1x8x4xi8>
!vecC = vector<1x8xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d1, d2)>
func.func @negative_brgemm_dimension(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_brgemm_dimension
// CHECK-NOT: x86.avx.dot.i8
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x16x2xbf16>
!vecC = vector<1x1x16xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_float_acc_type(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_float_acc_type
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xi8>
!vecB = vector<1x1x8x4xi8>
!vecC = vector<1x1x8xi8>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_int_acc_type(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_int_acc_type
// CHECK-NOT: x86.avx.dot.i8
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xbf16>
!vecB = vector<1x1x16x4xbf16>
!vecC = vector<1x1x16xbf16>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_wrong_vnni_blocking_factor_bf16(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_wrong_vnni_blocking_factor_bf16
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1xbf16>
!vecB = vector<1x1x32xbf16>
!vecC = vector<1x32xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @negative_brgemm_not_vnni(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_brgemm_not_vnni
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x4xi8>
!vecB = vector<1x1x32x4xi8>
!vecC = vector<1x1x32xi32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_wrong_vector_shape_int8(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_wrong_vector_shape_int8
// CHECK-NOT: x86.avx.dot.i8
// CHECK-NOT: x86.avx10.dot.i8
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1x1x2xbf16>
!vecB = vector<1x1x32x2xbf16>
!vecC = vector<1x1x32xf32>
#map = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d4, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_wrong_vector_shape_bf16(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_wrong_vector_shape_bf16
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x4xbf16>
!vecB = vector<4x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x4xbf16>
!memrefB = memref<4x32xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_flat_other_dim_is_not_2(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %3 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_flat_other_dim_is_not_2
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_flat_offset_diff_is_not16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %3 = vector.transfer_read %arg1[%c0, %c32], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_flat_offset_diff_is_not16
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_flat_dynamic_offset(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC, %arg3: index) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %3 = vector.transfer_read %arg1[%c0, %arg3], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %5 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_flat_dynamic_offset
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x2xbf16>
!vecB = vector<2x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x32xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_flat_read_after_contract(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %4 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC      

  %3 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB      
  %5 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_flat_read_after_contract
// CHECK-NOT: x86.avx512.dot
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1xbf16>
!vecB = vector<1x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_contracts_not_in_order(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c32] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c16] :
        !memrefC, !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c16] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_contracts_not_in_order
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.contract
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.store

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1xbf16>
!vecB = vector<1x32xbf16>
!vecC = vector<1x32xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_dim_is_32(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c32] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c32] :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c32] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_dim_is_32
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.contract
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.store

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1xbf16>
!vecB = vector<1x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_offset_diff_is_32(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %c32] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c16] :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c16] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_offset_diff_is_32
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.contract
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.store

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x1xbf16>
!vecB = vector<1x16xbf16>
!vecC = vector<1x16xf32>
!memrefA = memref<4x2xbf16>
!memrefB = memref<2x64xbf16>
!memrefC = memref<2x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_dynamic_offset(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC, %arg3: index) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : bf16
  %32 = ub.poison : f32
  %1 = vector.load %arg0[%c0, %c0] :
        !memrefA, !vecA
  %2 = vector.load %arg1[%c0, %c0] :
        !memrefB, !vecB
  %3 = vector.load %arg1[%c0, %arg3] :
        !memrefB, !vecB
  %4 = vector.load %arg2[%c0, %c0] :
        !memrefC, !vecC
  %5 = vector.load %arg2[%c0, %c16] :
        !memrefC, !vecC

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %2, %4
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %3, %5
    : !vecA, !vecB into !vecC

  vector.store %6, %arg2[%c0, %c0] : !memrefC, !vecC
  vector.store %7, %arg2[%c0, %c16] : !memrefC, !vecC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_dynamic_offset
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.contract
// CHECK-NOT: vector.shuffle
// CHECK-NOT: vector.shuffle
// CHECK: vector.store

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}

// -----

!vecA = vector<1x4xi8>
!vecB = vector<4x8xi8>
!vecC = vector<1x8xi32>
!memrefA = memref<4x4xi8>
!memrefB = memref<4x64xi8>
!memrefC = memref<4x64xi32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0,  d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0,  d1, d2) -> (d0, d1)>
func.func @negative_i8_avx2dp_flat_layout_offset_diff_16(
  %arg0: !memrefA, %arg1: !memrefB, %arg2: !memrefC) -> !memrefC
{
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %0 = ub.poison : i8
  %32 = ub.poison : i32
  %1 = vector.transfer_read %arg0[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefA, !vecA
  %2 = vector.transfer_read %arg2[%c0, %c0], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %3 = vector.transfer_read %arg2[%c0, %c16], %32 {in_bounds = [true, true]} :
        !memrefC, !vecC
  %4 = vector.transfer_read %arg1[%c0, %c0], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB
  %5 = vector.transfer_read %arg1[%c0, %c16], %0 {in_bounds = [true, true]} :
        !memrefB, !vecB

  %6 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %4, %2
    : !vecA, !vecB into !vecC

  %7 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %1, %5, %3
    : !vecA, !vecB into !vecC

  vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : !vecC, !memrefC
  vector.transfer_write %7, %arg2[%c0, %c16] {in_bounds = [true, true]} : !vecC, !memrefC

  return %arg2 : !memrefC
}

// CHECK-LABEL: @negative_i8_avx2dp_flat_layout_offset_diff_16
// CHECK-NOT: x86.avx.dot.i8
// CHECK: vector.contract

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.x86.vector_contract_to_packed_type_dot_product
    } : !transform.any_op
    transform.yield
  }
}
