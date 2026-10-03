// RUN: mlir-opt %s -test-single-fold -split-input-file | FileCheck %s
// RUN: mlir-opt %s -sccp -split-input-file | FileCheck %s

// These tests check the delinearize_index fold alone. -canonicalize also runs
// DropUnitExtentBasis, which would hide a broken fold.

// A unit basis element makes its result 0. The op stays for the other results.

// CHECK-LABEL: func @delinearize_unit_basis(
//  CHECK-SAME:     %[[IDX:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D:.+]]:3 = affine.delinearize_index %[[IDX]] into (%[[B]], 1, 4) : index, index, index
//       CHECK:   return %[[D]]#0, %[[C0]], %[[D]]#2 : index, index, index
func.func @delinearize_unit_basis(%idx: index, %b: index) -> (index, index, index) {
  %0:3 = affine.delinearize_index %idx into (%b, 1, 4) : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// An outer bound of 1 makes the first result 0.

// CHECK-LABEL: func @delinearize_unit_outer_bound(
//  CHECK-SAME:     %[[IDX:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D:.+]]:2 = affine.delinearize_index %[[IDX]] into (1, 8) : index, index
//       CHECK:   return %[[C0]], %[[D]]#1 : index, index
func.func @delinearize_unit_outer_bound(%idx: index) -> (index, index) {
  %0:2 = affine.delinearize_index %idx into (1, 8) : index, index
  return %0#0, %0#1 : index, index
}

// -----

// Without an outer bound, the first result has no bound. The first basis
// element of 1 applies to the second result.

// CHECK-LABEL: func @delinearize_unit_basis_no_outer_bound(
//  CHECK-SAME:     %[[IDX:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D:.+]]:3 = affine.delinearize_index %[[IDX]] into (1, 8) : index, index, index
//       CHECK:   return %[[D]]#0, %[[C0]], %[[D]]#2 : index, index, index
func.func @delinearize_unit_basis_no_outer_bound(%idx: index) -> (index, index, index) {
  %0:3 = affine.delinearize_index %idx into (1, 8) : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}

// -----

// CHECK-LABEL: func @delinearize_unit_basis_vector(
//  CHECK-SAME:     %[[VEC:[a-zA-Z0-9]+]]: vector<4xindex>)
//   CHECK-DAG:   %[[ZERO:.+]] = arith.constant dense<0> : vector<4xindex>
//   CHECK-DAG:   %[[D:.+]]:3 = affine.delinearize_index %[[VEC]] into (4, 1, 8) : vector<4xindex>, vector<4xindex>, vector<4xindex>
//       CHECK:   return %[[D]]#0, %[[ZERO]], %[[D]]#2 : vector<4xindex>, vector<4xindex>, vector<4xindex>
func.func @delinearize_unit_basis_vector(%vec: vector<4xindex>) -> (vector<4xindex>, vector<4xindex>, vector<4xindex>) {
  %0:3 = affine.delinearize_index %vec into (4, 1, 8) : vector<4xindex>, vector<4xindex>, vector<4xindex>
  return %0#0, %0#1, %0#2 : vector<4xindex>, vector<4xindex>, vector<4xindex>
}

// -----

// Without a unit basis element, nothing folds.

// CHECK-LABEL: func @delinearize_no_unit_basis(
//  CHECK-SAME:     %[[IDX:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[B:[a-zA-Z0-9]+]]: index)
//   CHECK-NOT:   arith.constant
//       CHECK:   %[[D:.+]]:3 = affine.delinearize_index %[[IDX]] into (%[[B]], 2, 4) : index, index, index
//       CHECK:   return %[[D]]#0, %[[D]]#1, %[[D]]#2 : index, index, index
func.func @delinearize_no_unit_basis(%idx: index, %b: index) -> (index, index, index) {
  %0:3 = affine.delinearize_index %idx into (%b, 2, 4) : index, index, index
  return %0#0, %0#1, %0#2 : index, index, index
}
