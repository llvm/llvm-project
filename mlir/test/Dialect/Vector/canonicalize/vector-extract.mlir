// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file -allow-unregistered-dialect | FileCheck %s

// Tests for FoldExtractFromInsertUnitDim: fold vector.extract from
// vector.insert when the extract position is a strict prefix of the insert
// position and all remaining dimensions are size 1, so the extracted
// sub-vector is fully determined by the inserted value.

// Basic case: extract row from a vector<4x1xf32> insert chain.
// The extract at [i] from insert at [i, 0] should fold to a broadcast.

// CHECK-LABEL: func.func @extract_from_insert_trailing_unit_dim
// CHECK-SAME:    %[[S0:.*]]: f32, %[[S1:.*]]: f32
// CHECK-NOT:     ub.poison
// CHECK-NOT:     vector.insert {{.*}} vector<4x1xf32>
// CHECK-NOT:     vector.extract {{.*}} vector<4x1xf32>
// CHECK-DAG:     vector.broadcast %[[S0]] : f32 to vector<1xf32>
// CHECK-DAG:     vector.broadcast %[[S1]] : f32 to vector<1xf32>
func.func @extract_from_insert_trailing_unit_dim(%s0: f32, %s1: f32) -> (vector<1xf32>, vector<1xf32>) {
  %poison = ub.poison : vector<4x1xf32>
  %ins0 = vector.insert %s0, %poison [0, 0] : f32 into vector<4x1xf32>
  %ins1 = vector.insert %s1, %ins0 [1, 0] : f32 into vector<4x1xf32>
  %ext0 = vector.extract %ins0 [0] : vector<1xf32> from vector<4x1xf32>
  %ext1 = vector.extract %ins1 [1] : vector<1xf32> from vector<4x1xf32>
  return %ext0, %ext1 : vector<1xf32>, vector<1xf32>
}

// -----

// Multiple trailing unit dims: vector<4x1x1xf32>.
// Extract at [i] gives vector<1x1xf32>; the inserted value fully
// determines the result.

// CHECK-LABEL: func.func @extract_from_insert_multiple_trailing_unit_dims
// CHECK-SAME:    %[[S:.*]]: f32
// CHECK-NOT:     ub.poison
// CHECK:         vector.broadcast %[[S]] : f32 to vector<1x1xf32>
func.func @extract_from_insert_multiple_trailing_unit_dims(%s: f32) -> vector<1x1xf32> {
  %poison = ub.poison : vector<4x1x1xf32>
  %ins = vector.insert %s, %poison [2, 0, 0] : f32 into vector<4x1x1xf32>
  %ext = vector.extract %ins [2] : vector<1x1xf32> from vector<4x1x1xf32>
  return %ext : vector<1x1xf32>
}

// -----

// Negative: extract position does not match insert position.
// The upstream extract fold forwards through to the poison dest,
// but our pattern should not fire.

// CHECK-LABEL: func.func @extract_from_insert_position_mismatch
// CHECK-NOT:     vector.broadcast
func.func @extract_from_insert_position_mismatch(%s: f32) -> vector<1xf32> {
  %poison = ub.poison : vector<4x1xf32>
  %ins = vector.insert %s, %poison [1, 0] : f32 into vector<4x1xf32>
  %ext = vector.extract %ins [0] : vector<1xf32> from vector<4x1xf32>
  return %ext : vector<1xf32>
}

// -----

// Negative: trailing dim is not unit size -- should NOT fold.

// CHECK-LABEL: func.func @extract_from_insert_non_unit_trailing_dim
// CHECK:         vector.insert
// CHECK:         vector.extract
func.func @extract_from_insert_non_unit_trailing_dim(%s: f32) -> vector<4xf32> {
  %poison = ub.poison : vector<3x4xf32>
  %ins = vector.insert %s, %poison [1, 2] : f32 into vector<3x4xf32>
  %ext = vector.extract %ins [1] : vector<4xf32> from vector<3x4xf32>
  return %ext : vector<4xf32>
}

// -----

// Negative: dynamic extract position -- should NOT fold.

// CHECK-LABEL: func.func @extract_from_insert_dynamic_position
// CHECK:         vector.insert
// CHECK:         vector.extract
func.func @extract_from_insert_dynamic_position(%s: f32, %idx: index) -> vector<1xf32> {
  %poison = ub.poison : vector<4x1xf32>
  %ins = vector.insert %s, %poison [2, 0] : f32 into vector<4x1xf32>
  %ext = vector.extract %ins [%idx] : vector<1xf32> from vector<4x1xf32>
  return %ext : vector<1xf32>
}

// -----

// Negative: dynamic insert position -- should NOT fold.

// CHECK-LABEL: func.func @insert_dynamic_position_not_folded
// CHECK:         vector.insert
// CHECK:         vector.extract
func.func @insert_dynamic_position_not_folded(%s: f32, %idx: index) -> vector<1xf32> {
  %poison = ub.poison : vector<4x1xf32>
  %ins = vector.insert %s, %poison [%idx, 0] : f32 into vector<4x1xf32>
  %ext = vector.extract %ins [2] : vector<1xf32> from vector<4x1xf32>
  return %ext : vector<1xf32>
}

// -----

// Exact position match (same number of indices) is handled by the
// existing extract fold, not our pattern. Verify it still works.

// CHECK-LABEL: func.func @extract_from_insert_exact_match
// CHECK-SAME:    %[[S:.*]]: f32
// CHECK:         return %[[S]]
func.func @extract_from_insert_exact_match(%s: f32) -> f32 {
  %poison = ub.poison : vector<4x1xf32>
  %ins = vector.insert %s, %poison [2, 0] : f32 into vector<4x1xf32>
  %ext = vector.extract %ins [2, 0] : f32 from vector<4x1xf32>
  return %ext : f32
}

// -----

// End-to-end: full insert chain with all extracts reading from intermediates.
// This mirrors the pattern from conv_1d lowering that triggered the original
// bug. After folding, the insert chain becomes dead and is eliminated.

// CHECK-LABEL: func.func @full_insert_chain_with_extracts
// CHECK-SAME:    %[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %[[D:.*]]: f32
// CHECK-NOT:     ub.poison : vector<4x1xf32>
// CHECK-NOT:     vector<4x1xf32>
// CHECK-DAG:     %[[BA:.*]] = vector.broadcast %[[A]] : f32 to vector<1xf32>
// CHECK-DAG:     %[[BB:.*]] = vector.broadcast %[[B]] : f32 to vector<1xf32>
// CHECK-DAG:     %[[BC:.*]] = vector.broadcast %[[C]] : f32 to vector<1xf32>
// CHECK-DAG:     %[[BD:.*]] = vector.broadcast %[[D]] : f32 to vector<1xf32>
// CHECK:         return %[[BA]], %[[BB]], %[[BC]], %[[BD]]
func.func @full_insert_chain_with_extracts(%a: f32, %b: f32, %c: f32, %d: f32) -> (vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>) {
  %poison = ub.poison : vector<4x1xf32>
  %ins0 = vector.insert %a, %poison [0, 0] : f32 into vector<4x1xf32>
  %ins1 = vector.insert %b, %ins0 [1, 0] : f32 into vector<4x1xf32>
  %ins2 = vector.insert %c, %ins1 [2, 0] : f32 into vector<4x1xf32>
  %ins3 = vector.insert %d, %ins2 [3, 0] : f32 into vector<4x1xf32>
  %ext0 = vector.extract %ins0 [0] : vector<1xf32> from vector<4x1xf32>
  %ext1 = vector.extract %ins1 [1] : vector<1xf32> from vector<4x1xf32>
  %ext2 = vector.extract %ins2 [2] : vector<1xf32> from vector<4x1xf32>
  %ext3 = vector.extract %ins3 [3] : vector<1xf32> from vector<4x1xf32>
  return %ext0, %ext1, %ext2, %ext3 : vector<1xf32>, vector<1xf32>, vector<1xf32>, vector<1xf32>
}
