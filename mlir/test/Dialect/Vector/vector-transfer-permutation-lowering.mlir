// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// vector.transfer_write -> vector.transpose + vector.transfer_write
/// [Pattern: TransferWritePermutationLowering]
///----------------------------------------------------------------------------------------
/// Input:
///   * vector.transfer_write op with a permutation that under a transpose
///     _would be_ a minor identity permutation map
/// Output:
///   * vector.transpose + vector.transfer_write with a permutation map which
///     _is_ a minor identity

// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map
// CHECK-SAME:      %[[VEC:.*]]: vector<4x8xi16>,
// CHECK-SAME:      %[[MEM:.*]]: memref<2x2x8x4xi16>
// CHECK:           %[[TR:.*]] = vector.transpose %[[VEC]], [1, 0] : vector<4x8xi16> to vector<8x4xi16>
// CHECK:           vector.transfer_write
// CHECK-NOT:       permutation_map
// CHECK-SAME:      %[[TR]], %[[MEM]]{{.*}} {in_bounds = [true, true]} : vector<8x4xi16>, memref<2x2x8x4xi16>
func.func @xfer_write_transposing_permutation_map(
    %vec: vector<4x8xi16>,
    %mem: memref<2x2x8x4xi16>,
    %idx: index) {

  vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx] {
    in_bounds = [true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x8xi16>, memref<2x2x8x4xi16>

  return
}

// Even with out-of-bounds accesses, it is safe to apply this pattern

// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map_out_of_bounds
// CHECK-SAME:      %[[VEC:.*]]: vector<4x8xi16>,
// CHECK-SAME:      %[[MEM:.*]]: memref<2x2x?x?xi16>,
// CHECK-SAME:      %[[IDX:.*]]: index) {
// CHECK:           %[[TR:.*]] = vector.transpose %[[VEC]], [1, 0] : vector<4x8xi16> to vector<8x4xi16>
// Expect the in_bounds attribute to be preserved. Since we don't print it when
// all flags are "false", it should not appear in the output.
// CHECK-NOT:       in_bounds
// CHECK:           vector.transfer_write
// CHECK-NOT:       permutation_map
// CHECK-SAME:      %[[TR]], %[[MEM]][%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]] : vector<8x4xi16>, memref<2x2x?x?xi16>
func.func @xfer_write_transposing_permutation_map_out_of_bounds(
    %vec: vector<4x8xi16>,
    %mem: memref<2x2x?x?xi16>,
    %idx: index) {

  vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx] {
    in_bounds = [false, false],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x8xi16>, memref<2x2x?x?xi16>

  return
}

// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map_with_mask_scalable
// CHECK-SAME:      %[[VEC:.*]]: vector<4x[8]xi16>,
// CHECK-SAME:      %[[MEM:.*]]: memref<2x2x?x4xi16>,
// CHECK-SAME:      %[[MASK:.*]]: vector<[8]x4xi1>
// CHECK:           %[[TR:.*]] = vector.transpose %[[VEC]], [1, 0] : vector<4x[8]xi16> to vector<[8]x4xi16>
// CHECK:           vector.transfer_write
// CHECK-NOT:       permutation_map
// CHECK-SAME:      %[[TR]], %[[MEM]]{{.*}}, %[[MASK]] {in_bounds = [true, true]} : vector<[8]x4xi16>, memref<2x2x?x4xi16>
func.func @xfer_write_transposing_permutation_map_with_mask_scalable(
    %vec: vector<4x[8]xi16>,
    %mem: memref<2x2x?x4xi16>,
    %mask: vector<[8]x4xi1>,
    %idx: index) {

  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx], %mask {
    in_bounds = [true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x[8]xi16>, memref<2x2x?x4xi16>

  return
}

// Masked version is not supported

// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map_masked
// CHECK-NOT: vector.transpose
func.func @xfer_write_transposing_permutation_map_masked(
    %vec: vector<4x8xi16>,
    %mem: memref<2x2x8x4xi16>,
    %mask: vector<8x4xi1>,
    %idx: index) {

  %c0 = arith.constant 0 : index
  vector.mask %mask {
    vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx] {
      in_bounds = [true, true],
      permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
    } : vector<4x8xi16>, memref<2x2x8x4xi16>
  } : vector<8x4xi1>

  return
}

///----------------------------------------------------------------------------------------
/// vector.transfer_write -> vector.broadcast + vector.transpose + vector.transfer_write
/// [Patterns: TransferWriteNonPermutationLowering + TransferWritePermutationLowering]
///----------------------------------------------------------------------------------------
/// Input:
///   * vector.transfer_write op with a map which _is not_ a permutation of a
///     minor identity
/// Output:
///   * vector.broadcast + vector.transpose + vector.transfer_write with a map
///     which _is_ a permutation of a minor identity

// CHECK-LABEL:   func.func @xfer_write_non_transposing_permutation_map(
// CHECK-SAME:      %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[VEC:.*]]: vector<7xf32>,
// CHECK-SAME:      %[[IDX_1:.*]]: index, %[[IDX_2:.*]]: index) {
// CHECK:           %[[BC:.*]] = vector.broadcast %[[VEC]] : vector<7xf32> to vector<1x7xf32>
// CHECK:           %[[TR:.*]] = vector.transpose %[[BC]], [1, 0] : vector<1x7xf32> to vector<7x1xf32>
// CHECK:           vector.transfer_write %[[TR]], %[[MEM]]{{\[}}%[[IDX_1]], %[[IDX_2]]] {in_bounds = [false, true]} : vector<7x1xf32>, memref<?x?xf32>
func.func @xfer_write_non_transposing_permutation_map(
    %mem : memref<?x?xf32>,
    %vec : vector<7xf32>,
    %idx_1 : index,
    %idx_2 : index) {

  vector.transfer_write %vec, %mem[%idx_1, %idx_2] {
    permutation_map = affine_map<(d0, d1) -> (d0)>
  } : vector<7xf32>, memref<?x?xf32>

  return
}

// Even with out-of-bounds accesses, it is safe to apply this pattern

// CHECK-LABEL:   func.func @xfer_write_non_transposing_permutation_map_with_mask_out_of_bounds(
// CHECK-SAME:      %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[VEC:.*]]: vector<7xf32>,
// CHECK-SAME:      %[[IDX_1:.*]]: index, %[[IDX_2:.*]]: index,
// CHECK-SAME:      %[[MASK:.*]]: vector<7xi1>) {
// CHECK:           %[[BC_VEC:.*]] = vector.broadcast %[[VEC]] : vector<7xf32> to vector<1x7xf32>
// CHECK:           %[[BC_MASK:.*]] = vector.broadcast %[[MASK]] : vector<7xi1> to vector<1x7xi1>
// CHECK:           %[[TR_MASK:.*]] = vector.transpose %[[BC_MASK]], [1, 0] : vector<1x7xi1> to vector<7x1xi1>
// CHECK:           %[[TR_VEC:.*]] = vector.transpose %[[BC_VEC]], [1, 0] : vector<1x7xf32> to vector<7x1xf32>
// CHECK:           vector.transfer_write %[[TR_VEC]], %[[MEM]]{{\[}}%[[IDX_1]], %[[IDX_2]]], %[[TR_MASK]] {in_bounds = [false, true]} : vector<7x1xf32>, memref<?x?xf32>
func.func @xfer_write_non_transposing_permutation_map_with_mask_out_of_bounds(
    %mem : memref<?x?xf32>,
    %vec : vector<7xf32>,
    %idx_1 : index,
    %idx_2 : index,
    %mask : vector<7xi1>) {

  vector.transfer_write %vec, %mem[%idx_1, %idx_2], %mask {
    permutation_map = affine_map<(d0, d1) -> (d0)>,
    in_bounds = [false]
  } : vector<7xf32>, memref<?x?xf32>

  return
}

// CHECK:           func.func @permutation_with_mask_xfer_write_scalable(
// CHECK-SAME:        %[[VEC:.*]]: vector<4x[8]xi16>,
// CHECK-SAME:        %[[MEM:.*]]: memref<1x4x?x1xi16>,
// CHECK-SAME:        %[[MASK:.*]]: vector<4x[8]xi1>
// CHECK:             %[[BC_1:.*]] = vector.broadcast %[[VEC]] : vector<4x[8]xi16> to vector<1x4x[8]xi16>
// CHECK:             %[[BC_2:.*]] = vector.broadcast %[[MASK]] : vector<4x[8]xi1> to vector<1x4x[8]xi1>
// CHECK:             %[[TRANSPOSE_1:.*]] =  vector.transpose %[[BC_2]], [1, 2, 0] : vector<1x4x[8]xi1> to vector<4x[8]x1xi1>
// CHECK:             %[[TRANSPOSE_2:.*]] =  vector.transpose %[[BC_1]], [1, 2, 0] : vector<1x4x[8]xi16> to vector<4x[8]x1xi16>
// CHECK:             vector.transfer_write %[[TRANSPOSE_2]], %[[MEM]]{{.*}}, %[[TRANSPOSE_1]] {in_bounds = [true, true, true]} : vector<4x[8]x1xi16>, memref<1x4x?x1xi16>
func.func @permutation_with_mask_xfer_write_scalable(
    %vec: vector<4x[8]xi16>,
    %mem: memref<1x4x?x1xi16>,
    %mask: vector<4x[8]xi1>,
    %idx: index){

  vector.transfer_write %vec, %mem[%idx, %idx, %idx, %idx], %mask {
    in_bounds = [true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
  } : vector<4x[8]xi16>, memref<1x4x?x1xi16>

  return
}

// Masked version is not supported

// CHECK-LABEL: func @masked_permutation_xfer_write_fixed_width
//  CHECK-SAME:   %[[DEST:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[VEC:.*]]: vector<16xf32>,
//  CHECK-SAME:   %[[IDX:.*]]: index,
//  CHECK-SAME:   %[[MASK:.*]]: vector<16xi1>
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.mask %[[MASK]] { vector.transfer_write %[[VEC]], %[[DEST]]{{.*}} vector<16xf32>, tensor<?x?xf32> } : vector<16xi1> -> tensor<?x?xf32>
func.func @masked_permutation_xfer_write_fixed_width(
    %dest: tensor<?x?xf32>,
    %vec: vector<16xf32>,
    %idx: index,
    %mask: vector<16xi1>) -> tensor<?x?xf32> {

  %res = vector.mask %mask {
    vector.transfer_write %vec, %dest[%idx, %idx] {
      permutation_map = affine_map<(d0, d1) -> (d0)>
    } : vector<16xf32>, tensor<?x?xf32>
  } : vector<16xi1> -> tensor<?x?xf32>

  return %res : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @masked_permutation_xfer_write_scalable(
//  CHECK-SAME:   %[[VEC:.*]]: vector<4x[8]xi16>,
//  CHECK-SAME:   %[[DEST:.*]]: tensor<?x?x?x?xf32>,
//  CHECK-SAME:   %[[MASK:.*]]: vector<4x[8]xi1>
//  CHECK-SAME:   -> tensor<?x?x?x?xf32> {
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.mask %[[MASK]] { vector.transfer_write %[[VEC]], %[[DEST]]{{.*}} : vector<4x[8]xi16>, tensor<?x?x?x?xf32> } : vector<4x[8]xi1> -> tensor<?x?x?x?xf32>
func.func @masked_permutation_xfer_write_scalable(
    %vec: vector<4x[8]xi16>,
    %dest: tensor<?x?x?x?xf32>,
    %mask:  vector<4x[8]xi1>,
    %idx: index) -> tensor<?x?x?x?xf32> {

  %c0 = arith.constant 0 : index
  %res = vector.mask %mask {
    vector.transfer_write %vec, %dest[%idx, %idx, %idx, %idx] {
      in_bounds = [true, true],
      permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
    } : vector<4x[8]xi16>, tensor<?x?x?x?xf32>
  } : vector<4x[8]xi1> -> tensor<?x?x?x?xf32>

  return %res : tensor<?x?x?x?xf32>
}

// Masked version is not supported

// CHECK-LABEL: func @masked_non_permutation_xfer_write_fixed_width
//  CHECK-SAME:   %[[DEST:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:   %[[VEC:.*]]: vector<14x8x16xf32>
//  CHECK-SAME:   %[[DIM:.*]]: index, %[[IDX:.*]]: index) -> tensor<?x?x?x?xf32>
//   CHECK-NOT:   vector.broadcast
//       CHECK:   vector.mask %0 { vector.transfer_write %[[VEC]], %[[DEST]]{{.*}} : vector<14x8x16xf32>, tensor<?x?x?x?xf32> } : vector<14x8x16xi1> -> tensor<?x?x?x?xf32>
func.func @masked_non_permutation_xfer_write_fixed_width(
    %dest : tensor<?x?x?x?xf32>,
    %vec : vector<14x8x16xf32>,
    %dim : index,
    %idx: index) -> tensor<?x?x?x?xf32> {

  %mask = vector.create_mask %dim, %dim, %dim : vector<14x8x16xi1>
  %res = vector.mask %mask {
    vector.transfer_write %vec, %dest[%idx, %idx, %idx, %idx] {
      in_bounds = [false, false, true],
      permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
    } : vector<14x8x16xf32>, tensor<?x?x?x?xf32>
  } : vector<14x8x16xi1> -> tensor<?x?x?x?xf32>

  return %res : tensor<?x?x?x?xf32>
}

///----------------------------------------------------------------------------------------
/// vector.transfer_read
///----------------------------------------------------------------------------------------
/// Input:
///   * vector.transfer_read op with a permutation map
/// Output:
///   * vector.transfer_read with a permutation map composed of leading zeros followed by a minor identiy +
///     vector.transpose op

// CHECK-LABEL:   func.func @permutation_with_mask_xfer_read_fixed_width(
// CHECK-SAME:      %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[DIM_1:.*]]: index, %[[DIM_2:.*]]: index, %[[IDX:.*]]: index) -> vector<8x4x2xf32> {
// CHECK:           %[[PASS_THROUGH:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_2]], %[[DIM_1]] : vector<2x4xi1>
// CHECK:           %[[T_READ:.*]] = vector.transfer_read %[[MEM]]{{\[}}%[[IDX]], %[[IDX]]], %[[PASS_THROUGH]], %[[MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[T_READ]] : vector<2x4xf32> to vector<8x2x4xf32>
// CHECK:           %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [0, 2, 1] : vector<8x2x4xf32> to vector<8x4x2xf32>
// CHECK:           return %[[TRANSPOSE]] : vector<8x4x2xf32>
func.func @permutation_with_mask_xfer_read_fixed_width(
    %mem: memref<?x?xf32>,
    %dim_1: index,
    %dim_2: index,
    %idx: index) -> (vector<8x4x2xf32>) {

  %pad = arith.constant 0.000000e+00 : f32

  %mask = vector.create_mask %dim_2, %dim_1 : vector<2x4xi1>
  %res = vector.transfer_read %mem[%idx, %idx], %pad, %mask {
    in_bounds = [true, true, true],
    permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>
  } : memref<?x?xf32>, vector<8x4x2xf32>

  return %res : vector<8x4x2xf32>
}

// CHECK-LABEL:   func.func @permutation_with_mask_xfer_read_scalable(
// CHECK-SAME:      %[[MEM:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[DIM_1:.*]]: index, %[[DIM_2:.*]]: index, %[[IDX:.*]]: index) -> vector<8x[4]x2xf32> {
// CHECK:           %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[DIM_2]], %[[DIM_1]] : vector<2x[4]xi1>
// CHECK:           %[[T_READ:.*]] = vector.transfer_read %[[MEM]]{{\[}}%[[IDX]], %[[IDX]]], %[[PAD]], %[[MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x[4]xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[T_READ]] : vector<2x[4]xf32> to vector<8x2x[4]xf32>
// CHECK:           %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [0, 2, 1] : vector<8x2x[4]xf32> to vector<8x[4]x2xf32>
// CHECK:           return %[[TRANSPOSE]] : vector<8x[4]x2xf32>
func.func @permutation_with_mask_xfer_read_scalable(
    %mem: memref<?x?xf32>,
    %dim_1: index,
    %dim_2: index,
    %idx: index) -> (vector<8x[4]x2xf32>) {

  %pad = arith.constant 0.000000e+00 : f32

  %mask = vector.create_mask %dim_2, %dim_1 : vector<2x[4]xi1>
  %res = vector.transfer_read %mem[%idx, %idx], %pad, %mask {
    in_bounds = [true, true, true],
    permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>
  } : memref<?x?xf32>, vector<8x[4]x2xf32>

  return %res : vector<8x[4]x2xf32>
}

// Masked version is not supported

// CHECK-LABEL: func @masked_permutation_xfer_read_fixed_width
//  CHECK-SAME:   %[[DEST:.*]]: tensor<?x1xf32>,
//  CHECK-SAME:   %[[MASK:.*]]: vector<4x1xi1>
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.mask %[[MASK]] { vector.transfer_read %[[DEST]]{{.*}}: tensor<?x1xf32>, vector<1x4x4xf32> } : vector<4x1xi1> -> vector<1x4x4xf32>
func.func @masked_permutation_xfer_read_fixed_width(
    %dest: tensor<?x1xf32>,
    %mask : vector<4x1xi1>,
    %idx: index) {

  %pad = arith.constant 0.000000e+00 : f32
  %3 = vector.mask %mask {
    vector.transfer_read %dest[%idx, %idx], %pad {
      permutation_map = affine_map<(d0, d1) -> (d1, 0, d0)>
    } : tensor<?x1xf32>, vector<1x4x4xf32>
  } : vector<4x1xi1> -> vector<1x4x4xf32>

  "test.some_use"(%3) : (vector<1x4x4xf32>) -> ()
  return
}

// CHECK-LABEL:  func.func @masked_permutation_xfer_read_scalable(
//  CHECK-SAME:    %[[DEST:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:    %[[MASK:.*]]: vector<2x[4]xi1>
//   CHECK-NOT:    vector.transpose
//       CHECK:    %[[T_READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[DEST]]{{.*}} : tensor<?x?xf32>, vector<8x[4]x2xf32> } : vector<2x[4]xi1> -> vector<8x[4]x2xf32>
func.func @masked_permutation_xfer_read_scalable(
  %dest: tensor<?x?xf32>,
  %mask : vector<2x[4]xi1>,
  %idx: index) -> vector<8x[4]x2xf32> {

  %pad = arith.constant 0.000000e+00 : f32

  %res = vector.mask %mask {
    vector.transfer_read %dest[%idx, %idx], %pad {
      in_bounds = [true, true, true],
      permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>
    } : tensor<?x?xf32>, vector<8x[4]x2xf32>
  } :vector<2x[4]xi1> -> vector<8x[4]x2xf32>

  return %res : vector<8x[4]x2xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// vector.transfer_read
///----------------------------------------------------------------------------------------
/// TODO: Review and categorize

//       CHECK:   #[[MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>
//       CHECK:   func.func @transfer_read_reduce_rank_scalable(
//  CHECK-SAME:     %[[MEM:.*]]: memref<?x?x?x?xf32>, %[[IDX:.*]]: index) -> vector<8x[4]x2x3xf32> {
//       CHECK:     %[[T_READ:.*]] = vector.transfer_read %[[MEM]][%[[IDX]], %[[IDX]], %[[IDX]], %[[IDX]]]{{.*}} permutation_map = #[[MAP]]} : memref<?x?x?x?xf32>, vector<[4]x2x3xf32>
//       CHECK:     %[[BC:.*]] = vector.broadcast %[[T_READ]] : vector<[4]x2x3xf32> to vector<8x[4]x2x3xf32>
//       CHECK:     return %[[BC]] : vector<8x[4]x2x3xf32>
func.func @transfer_read_reduce_rank_scalable(
    %mem: memref<?x?x?x?xf32>, %idx: index) -> vector<8x[4]x2x3xf32> {

  %pad = arith.constant 0.000000e+00 : f32

  %res = vector.transfer_read %mem[%idx, %idx, %idx, %idx], %pad {
    in_bounds = [true, true, true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d3)>
  } : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32>

  return %res : vector<8x[4]x2x3xf32>
}

// Masked version is not supported

// CHECK-LABEL:   func.func @masked_transfer_read_reduce_rank(
//  CHECK-SAME:     %[[MEM:.*]]: memref<?x?x?x?xf32>,
//  CHECK-SAME:     %[[DIM:.*]]: index,
//  CHECK-SAME:     %[[IDX:.*]]: index) -> vector<8x[4]x2x3xf32> {
//   CHECK-NOT:     vector.broadcast
//       CHECK:     %[[MASK:.*]] = vector.mask %0 { vector.transfer_read %[[MEM]]{{.*}} : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32> } : vector<[4]x3xi1> -> vector<8x[4]x2x3xf32>
func.func @masked_transfer_read_reduce_rank(
    %mem: memref<?x?x?x?xf32>,
    %dim: index,
    %idx: index) -> vector<8x[4]x2x3xf32> {

  %pad = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %dim, %dim: vector<[4]x3xi1>

  %res = vector.mask %mask {
    vector.transfer_read %mem[%idx, %idx, %idx, %idx], %pad {
      in_bounds = [true, true, true, true],
      permutation_map = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d3)>
    } : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32>
  } : vector<[4]x3xi1> -> vector<8x[4]x2x3xf32>

  return %res : vector<8x[4]x2x3xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
    transform.yield
  }
}
