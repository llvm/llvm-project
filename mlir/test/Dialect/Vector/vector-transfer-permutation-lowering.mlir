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
// CHECK-SAME:       %[[ARG_0:.*]]: vector<4x8xi16>,
// CHECK-SAME:       %[[MEM:.*]]: memref<2x2x8x4xi16>) {
// CHECK:           %[[TR:.*]] = vector.transpose %[[ARG_0]], [1, 0] : vector<4x8xi16> to vector<8x4xi16>
// CHECK:           vector.transfer_write
// CHECK-NOT:       permutation_map
// CHECK-SAME:      %[[TR]], %[[MEM]]{{.*}} {in_bounds = [true, true]} : vector<8x4xi16>, memref<2x2x8x4xi16>
func.func @xfer_write_transposing_permutation_map(
    %arg0: vector<4x8xi16>,
    %mem: memref<2x2x8x4xi16>) {

  %c0 = arith.constant 0 : index
  vector.transfer_write %arg0, %mem[%c0, %c0, %c0, %c0] {
    in_bounds = [true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x8xi16>, memref<2x2x8x4xi16>

  return
}

// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map_with_mask_scalable
// CHECK-SAME:      %[[ARG_0:.*]]: vector<4x[8]xi16>,
// CHECK-SAME:      %[[MEM:.*]]: memref<2x2x?x4xi16>,
// CHECK-SAME:      %[[MASK:.*]]: vector<[8]x4xi1>) {
// CHECK:           %[[TR:.*]] = vector.transpose %[[ARG_0]], [1, 0] : vector<4x[8]xi16> to vector<[8]x4xi16>
// CHECK:           vector.transfer_write
// CHECK-NOT:       permutation_map
// CHECK-SAME:      %[[TR]], %[[MEM]]{{.*}}, %[[MASK]] {in_bounds = [true, true]} : vector<[8]x4xi16>, memref<2x2x?x4xi16>
func.func @xfer_write_transposing_permutation_map_with_mask_scalable(
    %arg0: vector<4x[8]xi16>,
    %mem: memref<2x2x?x4xi16>,
    %mask: vector<[8]x4xi1>) {

  %c0 = arith.constant 0 : index
  vector.transfer_write %arg0, %mem[%c0, %c0, %c0, %c0], %mask {
    in_bounds = [true, true],
    permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
  } : vector<4x[8]xi16>, memref<2x2x?x4xi16>

  return
}

// Masked version is not supported
// CHECK-LABEL:   func.func @xfer_write_transposing_permutation_map_masked
// CHECK-NOT: vector.transpose
func.func @xfer_write_transposing_permutation_map_masked(
    %arg0: vector<4x8xi16>,
    %mem: memref<2x2x8x4xi16>,
    %mask: vector<8x4xi1>) {

  %c0 = arith.constant 0 : index
  vector.mask %mask {
    vector.transfer_write %arg0, %mem[%c0, %c0, %c0, %c0] {
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

// CHECK-LABEL: func @permutation_with_mask_xfer_write_fixed_width(
//       CHECK:   %[[vec:.*]] = arith.constant dense<-2.000000e+00> : vector<7x1xf32>
//       CHECK:   %[[mask:.*]] = arith.constant dense<[true, false, true, false, true, true, true]> : vector<7xi1>
//       CHECK:   %[[b:.*]] = vector.broadcast %[[mask]] : vector<7xi1> to vector<1x7xi1>
//       CHECK:   %[[tp:.*]] = vector.transpose %[[b]], [1, 0] : vector<1x7xi1> to vector<7x1xi1>
//       CHECK:   vector.transfer_write %[[vec]], %{{.*}}[%{{.*}}, %{{.*}}], %[[tp]] {in_bounds = [false, true]} : vector<7x1xf32>, memref<?x?xf32>
func.func @permutation_with_mask_xfer_write_fixed_width(%mem : memref<?x?xf32>, %base1 : index,
                                                   %base2 : index) {

  %fn1 = arith.constant -2.0 : f32
  %vf0 = vector.splat %fn1 : vector<7xf32>
  %mask = arith.constant dense<[1, 0, 1, 0, 1, 1, 1]> : vector<7xi1>
  vector.transfer_write %vf0, %mem[%base1, %base2], %mask
    {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [false]}
    : vector<7xf32>, memref<?x?xf32>
  return
}

// CHECK:           func.func @permutation_with_mask_xfer_write_scalable(
// CHECK-SAME:        %[[ARG_0:.*]]: vector<4x[8]xi16>,
// CHECK-SAME:        %[[ARG_1:.*]]: memref<1x4x?x1xi16>,
// CHECK-SAME:        %[[MASK:.*]]: vector<4x[8]xi1>) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[BCAST_1:.*]] = vector.broadcast %[[ARG_0]] : vector<4x[8]xi16> to vector<1x4x[8]xi16>
// CHECK:             %[[BCAST_2:.*]] = vector.broadcast %[[MASK]] : vector<4x[8]xi1> to vector<1x4x[8]xi1>
// CHECK:             %[[TRANSPOSE_1:.*]] =  vector.transpose %[[BCAST_2]], [1, 2, 0] : vector<1x4x[8]xi1> to vector<4x[8]x1xi1>
// CHECK:             %[[TRANSPOSE_2:.*]] =  vector.transpose %[[BCAST_1]], [1, 2, 0] : vector<1x4x[8]xi16> to vector<4x[8]x1xi16>
// CHECK:             vector.transfer_write %[[TRANSPOSE_2]], %[[ARG_1]]{{.*}}, %[[TRANSPOSE_1]] {in_bounds = [true, true, true]} : vector<4x[8]x1xi16>, memref<1x4x?x1xi16>
func.func @permutation_with_mask_xfer_write_scalable(%arg0: vector<4x[8]xi16>, %mem: memref<1x4x?x1xi16>, %mask:  vector<4x[8]xi1>){
     %c0 = arith.constant 0 : index
      vector.transfer_write %arg0, %mem[%c0, %c0, %c0, %c0], %mask {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
} : vector<4x[8]xi16>, memref<1x4x?x1xi16>

    return
}

// transfer_write in MaskOp case not supported.
// CHECK-LABEL: func @masked_permutation_xfer_write_fixed_width
//  CHECK-SAME:        %[[ARG_0:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:        %[[ARG_1:.*]]: vector<16xf32>,
//  CHECK-SAME:        %[[IDX:.*]]: index,
//  CHECK-SAME:        %[[MASK:.*]]: vector<16xi1>
//   CHECK-NOT:   vector.transpose
//       CHECK:   %[[RES:.*]] = vector.mask %[[MASK]] { vector.transfer_write %[[ARG_1]], %[[ARG_0]]{{.*}} vector<16xf32>, tensor<?x?xf32> } : vector<16xi1> -> tensor<?x?xf32>
func.func @masked_permutation_xfer_write_fixed_width(%t: tensor<?x?xf32>, %val: vector<16xf32>, %idx: index, %mask: vector<16xi1>) -> tensor<?x?xf32> {
  %r = vector.mask %mask { vector.transfer_write %val, %t[%idx, %idx] {permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<16xf32>, tensor<?x?xf32> } : vector<16xi1> -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @masked_permutation_xfer_write_scalable(
//  CHECK-SAME:        %[[ARG_0:.*]]: vector<4x[8]xi16>,
//  CHECK-SAME:        %[[ARG_1:.*]]: tensor<?x?x?x?xf32>,
//  CHECK-SAME:        %[[MASK:.*]]: vector<4x[8]xi1>)
//  CHECK-SAME:        -> tensor<?x?x?x?xf32> {
//   CHECK-NOT:   vector.transpose
//       CHECK:   %[[R:.*]] = vector.mask %[[MASK]] { vector.transfer_write %[[ARG_0]], %[[ARG_1]]{{.*}} : vector<4x[8]xi16>, tensor<?x?x?x?xf32> } : vector<4x[8]xi1> -> tensor<?x?x?x?xf32>
func.func @masked_permutation_xfer_write_scalable(%arg0: vector<4x[8]xi16>, %t: tensor<?x?x?x?xf32>, %mask:  vector<4x[8]xi1>) -> tensor<?x?x?x?xf32> {
     %c0 = arith.constant 0 : index
     %r = vector.mask %mask { vector.transfer_write %arg0, %t[%c0, %c0, %c0, %c0] {in_bounds = [true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
} : vector<4x[8]xi16>, tensor<?x?x?x?xf32> } : vector<4x[8]xi1> -> tensor<?x?x?x?xf32>

    return %r : tensor<?x?x?x?xf32>
}

// transfer_write in MaskOp case not supported.
// CHECK-LABEL: func @masked_non_permutation_xfer_write_fixed_width
//  CHECK-SAME:      %[[ARG0:.*]]: tensor<?x?x?x?xf32>
//  CHECK-SAME:      %[[ARG1:.*]]: vector<14x8x16xf32>
//  CHECK-SAME:      %[[IDX:.*]]: index) -> tensor<?x?x?x?xf32>
//   CHECK-NOT:   vector.broadcast
//       CHECK:   %[[masked1:.*]] = vector.mask %0 { vector.transfer_write %[[ARG1]], %[[ARG0]]{{.*}} : vector<14x8x16xf32>, tensor<?x?x?x?xf32> } : vector<14x8x16xi1> -> tensor<?x?x?x?xf32>
func.func @masked_non_permutation_xfer_write_fixed_width(
    %arg0 : tensor<?x?x?x?xf32>,
    %v1 : vector<14x8x16xf32>, %dim : index) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim, %dim, %dim : vector<14x8x16xi1>
  %0 = vector.mask %mask { vector.transfer_write %v1, %arg0[%c0, %c0, %c0, %c0] {in_bounds = [false, false, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>} : vector<14x8x16xf32>, tensor<?x?x?x?xf32> } : vector<14x8x16xi1> -> tensor<?x?x?x?xf32>

  return %0 : tensor<?x?x?x?xf32>
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
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDX_1:.*]]: index,
// CHECK-SAME:      %[[IDX_2:.*]]: index) -> vector<8x4x2xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PASS_THROUGH:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[IDX_2]], %[[IDX_1]] : vector<2x4xi1>
// CHECK:           %[[T_READ:.*]] = vector.transfer_read %[[ARG_0]]{{\[}}%[[C0]], %[[C0]]], %[[PASS_THROUGH]], %[[MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[T_READ]] : vector<2x4xf32> to vector<8x2x4xf32>
// CHECK:           %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [0, 2, 1] : vector<8x2x4xf32> to vector<8x4x2xf32>
// CHECK:           return %[[TRANSPOSE]] : vector<8x4x2xf32>
func.func @permutation_with_mask_xfer_read_fixed_width(%mem: memref<?x?xf32>, %dim_1: index, %dim_2: index) -> (vector<8x4x2xf32>) {

  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32

  %mask = vector.create_mask %dim_2, %dim_1 : vector<2x4xi1>
  %1 = vector.transfer_read %mem[%c0, %c0], %cst_0, %mask
    {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>}
    : memref<?x?xf32>, vector<8x4x2xf32>
  return %1 : vector<8x4x2xf32>
}

// CHECK-LABEL:   func.func @permutation_with_mask_xfer_read_scalable(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?x?xf32>,
// CHECK-SAME:      %[[IDX_1:.*]]: index,
// CHECK-SAME:      %[[IDX_2:.*]]: index) -> vector<8x[4]x2xf32> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[PASS_THROUGH:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[IDX_2]], %[[IDX_1]] : vector<2x[4]xi1>
// CHECK:           %[[T_READ:.*]] = vector.transfer_read %[[ARG_0]]{{\[}}%[[C0]], %[[C0]]], %[[PASS_THROUGH]], %[[MASK]] {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x[4]xf32>
// CHECK:           %[[BCAST:.*]] = vector.broadcast %[[T_READ]] : vector<2x[4]xf32> to vector<8x2x[4]xf32>
// CHECK:           %[[TRANSPOSE:.*]] = vector.transpose %[[BCAST]], [0, 2, 1] : vector<8x2x[4]xf32> to vector<8x[4]x2xf32>
// CHECK:           return %[[TRANSPOSE]] : vector<8x[4]x2xf32>
func.func @permutation_with_mask_xfer_read_scalable(%mem: memref<?x?xf32>, %dim_1: index, %dim_2: index) -> (vector<8x[4]x2xf32>) {

  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32

  %mask = vector.create_mask %dim_2, %dim_1 : vector<2x[4]xi1>
  %1 = vector.transfer_read %mem[%c0, %c0], %cst_0, %mask
    {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>}
    : memref<?x?xf32>, vector<8x[4]x2xf32>
  return %1 : vector<8x[4]x2xf32>
}

// transfer_read in MaskOp case not supported.
// CHECK-LABEL: func @masked_permutation_xfer_read_fixed_width
//  CHECK-SAME:        %[[ARG_0:.*]]: tensor<?x1xf32>,
//  CHECK-SAME:        %[[ARG_1:.*]]: vector<4x1xi1>
//   CHECK-NOT:   vector.transpose
//       CHECK:   vector.mask %[[ARG_1]] { vector.transfer_read %[[ARG_0]]{{.*}}: tensor<?x1xf32>, vector<1x4x4xf32> } : vector<4x1xi1> -> vector<1x4x4xf32>
func.func @masked_permutation_xfer_read_fixed_width(%arg0: tensor<?x1xf32>, %mask : vector<4x1xi1>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %3 = vector.mask %mask { vector.transfer_read %arg0[%c0, %c0], %cst {permutation_map = affine_map<(d0, d1) -> (d1, 0, d0)>} : tensor<?x1xf32>, vector<1x4x4xf32> } : vector<4x1xi1> -> vector<1x4x4xf32>
  call @test.some_use(%3) : (vector<1x4x4xf32>) -> ()
  return
}
func.func private @test.some_use(vector<1x4x4xf32>)

// CHECK-LABEL:  func.func @masked_permutation_xfer_read_scalable(
//  CHECK-SAME:      %[[ARG_0:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:      %[[MASK:.*]]: vector<2x[4]xi1>) -> vector<8x[4]x2xf32> {
//   CHECK-NOT:    vector.transpose
//       CHECK:    %[[T_READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[ARG_0]]{{.*}} : tensor<?x?xf32>, vector<8x[4]x2xf32> } : vector<2x[4]xi1> -> vector<8x[4]x2xf32>
func.func @masked_permutation_xfer_read_scalable(%t: tensor<?x?xf32>, %mask : vector<2x[4]xi1>) -> vector<8x[4]x2xf32> {

  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32

  %1 = vector.mask %mask { vector.transfer_read %t[%c0, %c0], %cst_0
    {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1) -> (0, d1, d0)>}
    : tensor<?x?xf32>, vector<8x[4]x2xf32> } :vector<2x[4]xi1> -> vector<8x[4]x2xf32>
  return %1 : vector<8x[4]x2xf32>
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
//  CHECK-SAME:       %[[ARG_0:.*]]: memref<?x?x?x?xf32>) -> vector<8x[4]x2x3xf32> {
//       CHECK:     %[[C0:.*]] = arith.constant 0 : index
//       CHECK:     %[[TFR:.*]] = vector.transfer_read %arg0[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]{{.*}} permutation_map = #[[MAP]]} : memref<?x?x?x?xf32>, vector<[4]x2x3xf32>
//       CHECK:     %[[BC:.*]] = vector.broadcast %[[TFR]] : vector<[4]x2x3xf32> to vector<8x[4]x2x3xf32>
//       CHECK:     return %[[BC]] : vector<8x[4]x2x3xf32>
func.func @transfer_read_reduce_rank_scalable(%mem: memref<?x?x?x?xf32>) -> vector<8x[4]x2x3xf32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %1 = vector.transfer_read %mem[%c0, %c0, %c0, %c0], %cst_0
    {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d3)>}
    : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32>
  return %1 : vector<8x[4]x2x3xf32>
}

// Masked case not supported.
// CHECK-LABEL:   func.func @masked_transfer_read_reduce_rank(
//  CHECK-SAME:       %[[ARG_0:.*]]: memref<?x?x?x?xf32>,
//  CHECK-SAME:       %[[DIM:.*]]: index) -> vector<8x[4]x2x3xf32> {
//   CHECK-NOT:     vector.broadcast
//       CHECK:     %[[MASK:.*]] = vector.mask %0 { vector.transfer_read %arg0{{.*}} : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32> } : vector<[4]x3xi1> -> vector<8x[4]x2x3xf32>
func.func @masked_transfer_read_reduce_rank(%mem: memref<?x?x?x?xf32>, %dim: index) -> vector<8x[4]x2x3xf32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %dim, %dim: vector<[4]x3xi1>
  %res = vector.mask %mask { vector.transfer_read %mem[%c0, %c0, %c0, %c0], %cst_0
    {in_bounds = [true, true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d3)>}
    : memref<?x?x?x?xf32>, vector<8x[4]x2x3xf32> } : vector<[4]x3xi1> -> vector<8x[4]x2x3xf32>
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
