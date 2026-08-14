// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// Tests for foldSize1TransferPermutationMap: when a transfer_read or
// transfer_write operates on a vector<1xT>, the permutation map is
// irrelevant and can be replaced with the minor identity map.

// +---------------------------------------------------------------------------
//  Tests of foldSize1TransferPermutationMap
// +---------------------------------------------------------------------------

// transfer_read with vector<1xf32> and non-minor-identity map: canonicalize
// the permutation map to minor identity.
//   (d0, d1) -> (d0) becomes (d0, d1) -> (d1)

// CHECK-LABEL: func.func @canonicalize_transfer_read_size1_map
//  CHECK-SAME:   (%[[M:.+]]: memref<4x3xf32>, %[[I:.+]]: index, %[[J:.+]]: index, %[[PAD:.+]]: f32)
//       CHECK:   %[[V:.+]] = vector.transfer_read %[[M]][%[[I]], %[[J]]], %[[PAD]]
//  CHECK-SAME:     {in_bounds = [true]}
//  CHECK-SAME:     : memref<4x3xf32>, vector<1xf32>
//   CHECK-NOT:     permutation_map
//       CHECK:   return %[[V]]
func.func @canonicalize_transfer_read_size1_map(%m: memref<4x3xf32>, %i: index, %j: index, %pad: f32) -> vector<1xf32> {
  %v = vector.transfer_read %m[%i, %j], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4x3xf32>, vector<1xf32>
  return %v : vector<1xf32>
}

// -----

// transfer_write with vector<1xf32> and non-minor-identity map: canonicalize.

// CHECK-LABEL: func.func @canonicalize_transfer_write_size1_map
//  CHECK-SAME:   (%[[V:.+]]: vector<1xf32>, %[[M:.+]]: memref<4x3xf32>, %[[I:.+]]: index, %[[J:.+]]: index)
//       CHECK:   vector.transfer_write %[[V]], %[[M]][%[[I]], %[[J]]]
//  CHECK-SAME:     {in_bounds = [true]}
//  CHECK-SAME:     : vector<1xf32>, memref<4x3xf32>
//   CHECK-NOT:     permutation_map
func.func @canonicalize_transfer_write_size1_map(%v: vector<1xf32>, %m: memref<4x3xf32>, %i: index, %j: index) {
  vector.transfer_write %v, %m[%i, %j] {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : vector<1xf32>, memref<4x3xf32>
  return
}

// -----

// Rank-3 memref: (d0, d1, d2) -> (d0) becomes (d0, d1, d2) -> (d2).

// CHECK-LABEL: func.func @canonicalize_transfer_read_size1_rank3
//       CHECK:   vector.transfer_read
//   CHECK-NOT:     permutation_map
//  CHECK-SAME:     : memref<4x3x2xf32>, vector<1xf32>
func.func @canonicalize_transfer_read_size1_rank3(%m: memref<4x3x2xf32>, %i: index, %j: index, %k: index, %pad: f32) -> vector<1xf32> {
  %v = vector.transfer_read %m[%i, %j, %k], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d0)>} : memref<4x3x2xf32>, vector<1xf32>
  return %v : vector<1xf32>
}

// -----

// Middle dimension map: (d0, d1, d2) -> (d1) becomes (d0, d1, d2) -> (d2).

// CHECK-LABEL: func.func @canonicalize_transfer_read_size1_middle_dim
//       CHECK:   vector.transfer_read
//   CHECK-NOT:     permutation_map
//  CHECK-SAME:     : memref<4x3x2xf32>, vector<1xf32>
func.func @canonicalize_transfer_read_size1_middle_dim(%m: memref<4x3x2xf32>, %i: index, %j: index, %k: index, %pad: f32) -> vector<1xf32> {
  %v = vector.transfer_read %m[%i, %j, %k], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1, d2) -> (d1)>} : memref<4x3x2xf32>, vector<1xf32>
  return %v : vector<1xf32>
}

// -----

// Negative: already minor identity -- should not be modified.

// CHECK-LABEL: func.func @negative_size1_already_minor_identity
//       CHECK:   vector.transfer_read
//   CHECK-NOT:     permutation_map
//  CHECK-SAME:     : memref<4x3xf32>, vector<1xf32>
func.func @negative_size1_already_minor_identity(%m: memref<4x3xf32>, %i: index, %j: index, %pad: f32) -> vector<1xf32> {
  %v = vector.transfer_read %m[%i, %j], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d1)>} : memref<4x3xf32>, vector<1xf32>
  return %v : vector<1xf32>
}

// -----

// Negative: vector<4xf32> (size != 1) -- should not be modified.

// CHECK-LABEL: func.func @negative_size1_size_not_one
//       CHECK:   vector.transfer_read
//  CHECK-SAME:     permutation_map
//  CHECK-SAME:     : memref<4x3xf32>, vector<4xf32>
func.func @negative_size1_size_not_one(%m: memref<4x3xf32>, %i: index, %j: index, %pad: f32) -> vector<4xf32> {
  %v = vector.transfer_read %m[%i, %j], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4x3xf32>, vector<4xf32>
  return %v : vector<4xf32>
}

// -----

// Negative: rank-0 vector<f32> -- should not be modified.

// CHECK-LABEL: func.func @negative_size1_rank0
//       CHECK:   vector.transfer_read
//  CHECK-SAME:     : memref<4xf32>, vector<f32>
func.func @negative_size1_rank0(%m: memref<4xf32>, %i: index, %pad: f32) -> vector<f32> {
  %v = vector.transfer_read %m[%i], %pad : memref<4xf32>, vector<f32>
  return %v : vector<f32>
}

// -----

// Negative: rank-2 vector -- should not be modified.

// CHECK-LABEL: func.func @negative_size1_rank2
//       CHECK:   vector.transfer_read
//  CHECK-SAME:     : memref<4x3xf32>, vector<2x4xf32>
func.func @negative_size1_rank2(%m: memref<4x3xf32>, %i: index, %j: index, %pad: f32) -> vector<2x4xf32> {
  %v = vector.transfer_read %m[%i, %j], %pad {in_bounds = [true, true]} : memref<4x3xf32>, vector<2x4xf32>
  return %v : vector<2x4xf32>
}

// -----

// Negative: scalable vector<[1]xf32> -- the runtime length is vscale, not 1,
// so the permutation map is semantically relevant.

// CHECK-LABEL: func.func @negative_size1_scalable
//       CHECK:   vector.transfer_read
//  CHECK-SAME:     permutation_map
//  CHECK-SAME:     : memref<4x3xf32>, vector<[1]xf32>
func.func @negative_size1_scalable(%m: memref<4x3xf32>, %i: index, %j: index, %pad: f32) -> vector<[1]xf32> {
  %v = vector.transfer_read %m[%i, %j], %pad {in_bounds = [true], permutation_map = affine_map<(d0, d1) -> (d0)>} : memref<4x3xf32>, vector<[1]xf32>
  return %v : vector<[1]xf32>
}
