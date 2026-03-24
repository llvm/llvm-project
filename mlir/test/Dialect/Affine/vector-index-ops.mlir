// RUN: mlir-opt %s -split-input-file --canonicalize | FileCheck %s --check-prefix=CANON
// RUN: mlir-opt %s -split-input-file --affine-expand-index-ops-as-affine | FileCheck %s --check-prefix=EXPAND

// Canonicalization: cancel disjoint linearize -> delinearize on vectors.

// CANON-LABEL: @cancel_linearize_delinearize_vector
// CANON-SAME:    (%[[V0:.+]]: vector<16xindex>, %[[V1:.+]]: vector<16xindex>)
// CANON:         return %[[V0]], %[[V1]]
func.func @cancel_linearize_delinearize_vector(%v0: vector<16xindex>, %v1: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>) {
  %0 = affine.linearize_index disjoint [%v0, %v1] by (4, 8) : vector<16xindex>
  %1:2 = affine.delinearize_index %0 into (4, 8) : vector<16xindex>, vector<16xindex>
  return %1#0, %1#1 : vector<16xindex>, vector<16xindex>
}

// -----
// Canonicalization: drop unit-extent basis on vector delinearize.

// CANON-LABEL: @drop_unit_extent_vector
// CANON-SAME:    (%[[VEC:.+]]: vector<16xindex>)
// CANON-DAG:     %[[ZERO:.+]] = arith.constant dense<0> : vector<16xindex>
// CANON:         %[[R:.+]]:2 = affine.delinearize_index %[[VEC]] into (4, 8) : vector<16xindex>, vector<16xindex>
// CANON:         return %[[R]]#0, %[[ZERO]], %[[R]]#1
func.func @drop_unit_extent_vector(%vec: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>, vector<16xindex>) {
  %0:3 = affine.delinearize_index %vec into (4, 1, 8) : vector<16xindex>, vector<16xindex>, vector<16xindex>
  return %0#0, %0#1, %0#2 : vector<16xindex>, vector<16xindex>, vector<16xindex>
}

// -----
// Canonicalization: drop unit-extent basis on vector linearize.

// CANON-LABEL: @drop_unit_linearize_vector
// CANON-SAME:    (%[[V0:.+]]: vector<8xindex>, %[[V1:.+]]: vector<8xindex>)
// CANON:         return %[[V0]]
func.func @drop_unit_linearize_vector(%v0: vector<8xindex>, %v1: vector<8xindex>) -> vector<8xindex> {
  %0 = affine.linearize_index disjoint [%v0, %v1] by (4, 1) : vector<8xindex>
  return %0 : vector<8xindex>
}

// -----
// Canonicalization: fold single-result vector delinearize to identity.

// CANON-LABEL: @fold_single_result_vector
// CANON-SAME:    (%[[VEC:.+]]: vector<4xindex>)
// CANON:         return %[[VEC]]
func.func @fold_single_result_vector(%vec: vector<4xindex>) -> vector<4xindex> {
  %0:1 = affine.delinearize_index %vec into () : vector<4xindex>
  return %0#0 : vector<4xindex>
}

// -----
// Canonicalization: fold single-index vector linearize to identity.

// CANON-LABEL: @fold_single_index_vector
// CANON-SAME:    (%[[VEC:.+]]: vector<4xindex>)
// CANON:         return %[[VEC]]
func.func @fold_single_index_vector(%vec: vector<4xindex>) -> vector<4xindex> {
  %0 = affine.linearize_index [%vec] by () : vector<4xindex>
  return %0 : vector<4xindex>
}

// -----
// Expansion: vector delinearize lowers to arith div/rem.

// EXPAND-LABEL: @expand_delinearize_vector
// EXPAND-SAME:    (%[[VEC:.+]]: vector<16xindex>)
// EXPAND-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<16xindex>
// EXPAND:         %[[DIV:.+]] = arith.divsi %[[VEC]], %[[C8]]
// EXPAND:         %[[MUL:.+]] = arith.muli %[[DIV]], %[[C8]]
// EXPAND:         %[[REM:.+]] = arith.subi %[[VEC]], %[[MUL]]
// EXPAND:         return %[[DIV]], %[[REM]]
func.func @expand_delinearize_vector(%vec: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>) {
  %0:2 = affine.delinearize_index %vec into (4, 8) : vector<16xindex>, vector<16xindex>
  return %0#0, %0#1 : vector<16xindex>, vector<16xindex>
}

// -----
// Expansion: vector linearize lowers to arith mul/add.

// EXPAND-LABEL: @expand_linearize_vector
// EXPAND-SAME:    (%[[V0:.+]]: vector<16xindex>, %[[V1:.+]]: vector<16xindex>)
// EXPAND-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<16xindex>
// EXPAND:         %[[MUL:.+]] = arith.muli %[[V0]], %[[C8]]
// EXPAND:         %[[ADD:.+]] = arith.addi %[[MUL]], %[[V1]]
// EXPAND:         return %[[ADD]]
func.func @expand_linearize_vector(%v0: vector<16xindex>, %v1: vector<16xindex>) -> vector<16xindex> {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<16xindex>
  return %0 : vector<16xindex>
}

// -----
// Expansion: 3D vector delinearize.

// EXPAND-LABEL: @expand_delinearize_vector_3d
// EXPAND-SAME:    (%[[VEC:.+]]: vector<16xindex>)
// EXPAND-DAG:     %[[C4:.+]] = arith.constant dense<4> : vector<16xindex>
// EXPAND-DAG:     %[[C12:.+]] = arith.constant dense<12> : vector<16xindex>
// EXPAND:         %[[D0:.+]] = arith.divsi %[[VEC]], %[[C12]]
// EXPAND:         %[[M0:.+]] = arith.muli %[[D0]], %[[C12]]
// EXPAND:         %[[R0:.+]] = arith.subi %[[VEC]], %[[M0]]
// EXPAND:         %[[D1:.+]] = arith.divsi %[[R0]], %[[C4]]
// EXPAND:         %[[M1:.+]] = arith.muli %[[D1]], %[[C4]]
// EXPAND:         %[[R1:.+]] = arith.subi %[[R0]], %[[M1]]
// EXPAND:         return %[[D0]], %[[D1]], %[[R1]]
func.func @expand_delinearize_vector_3d(%vec: vector<16xindex>) -> (vector<16xindex>, vector<16xindex>, vector<16xindex>) {
  %0:3 = affine.delinearize_index %vec into (2, 3, 4) : vector<16xindex>, vector<16xindex>, vector<16xindex>
  return %0#0, %0#1, %0#2 : vector<16xindex>, vector<16xindex>, vector<16xindex>
}

// -----
// Expansion: vector linearize -> offset -> delinearize pattern
// (as would be used in vector.gather lowering).

// EXPAND-LABEL: @vector_linearize_offset_delinearize
// EXPAND-SAME:    (%[[V0:.+]]: vector<4xindex>, %[[V1:.+]]: vector<4xindex>, %[[OFF:.+]]: vector<4xindex>)
// EXPAND-DAG:     %[[C8:.+]] = arith.constant dense<8> : vector<4xindex>
// EXPAND:         %[[LIN:.+]] = arith.muli %[[V0]], %[[C8]]
// EXPAND:         %[[LIN2:.+]] = arith.addi %[[LIN]], %[[V1]]
// EXPAND:         %[[FLAT:.+]] = arith.addi %[[LIN2]], %[[OFF]]
// EXPAND:         %[[DIV:.+]] = arith.divsi %[[FLAT]], %[[C8]]
// EXPAND:         %[[MUL:.+]] = arith.muli %[[DIV]], %[[C8]]
// EXPAND:         %[[REM:.+]] = arith.subi %[[FLAT]], %[[MUL]]
// EXPAND:         return %[[DIV]], %[[REM]]
func.func @vector_linearize_offset_delinearize(%v0: vector<4xindex>, %v1: vector<4xindex>, %offsets: vector<4xindex>) -> (vector<4xindex>, vector<4xindex>) {
  %0 = affine.linearize_index [%v0, %v1] by (4, 8) : vector<4xindex>
  %1 = arith.addi %0, %offsets : vector<4xindex>
  %2:2 = affine.delinearize_index %1 into (4, 8) : vector<4xindex>, vector<4xindex>
  return %2#0, %2#1 : vector<4xindex>, vector<4xindex>
}
