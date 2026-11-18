// RUN: tr-opt %s --tr-index-to-affine | FileCheck %s

// Milestone 14: only genuinely affine index relations become affine.apply.
// affine.for is used for a constant-bound loop; Linalg stays in scf.for.

// CHECK-DAG: #[[$ROWBASE:.*]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG: #[[$GROW:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
// CHECK-DAG: #[[$GCOL:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 32 + s1 + s2 * 128)>

// CHECK-LABEL: func.func @row_base
// CHECK-SAME: (%[[PID:.*]]: index)
func.func @row_base(%pid: index) -> index {
  %c128 = arith.constant 128 : index
  // CHECK: %[[B:.*]] = affine.apply #[[$ROWBASE]]()[%[[PID]]]
  // CHECK: return %[[B]]
  %rowBase = arith.muli %pid, %c128 : index
  return %rowBase : index
}

// CHECK-LABEL: func.func @global_row
// CHECK-SAME: (%[[PID:.*]]: index, %[[LOCAL:.*]]: index)
func.func @global_row(%pid: index, %local: index) -> index {
  %c128 = arith.constant 128 : index
  %rowBase = arith.muli %pid, %c128 : index
  // CHECK: %[[G:.*]] = affine.apply #[[$GROW]]()[%[[LOCAL]], %[[PID]]]
  // CHECK: return %[[G]]
  %globalRow = arith.addi %rowBase, %local : index
  return %globalRow : index
}

// CHECK-LABEL: func.func @global_col
// CHECK-SAME: (%[[KT:.*]]: index, %[[LANE:.*]]: index, %[[J:.*]]: index)
func.func @global_col(%kt: index, %lane: index, %j: index) -> index {
  %c128 = arith.constant 128 : index
  %c32 = arith.constant 32 : index
  %a = arith.muli %kt, %c128 : index
  %b = arith.addi %a, %lane : index
  %c = arith.muli %j, %c32 : index
  // CHECK: %[[G:.*]] = affine.apply #[[$GCOL]]()[%[[J]], %[[LANE]], %[[KT]]]
  // CHECK: return %[[G]]
  %globalCol = arith.addi %b, %c : index
  return %globalCol : index
}

// Product of two SSA values is not affine.
// CHECK-LABEL: func.func @not_affine
func.func @not_affine(%a: index, %b: index) -> index {
  // CHECK: arith.muli
  // CHECK-NOT: affine.apply
  %p = arith.muli %a, %b : index
  return %p : index
}

// affine.for is the natural form for a constant-bound local-row walk.
// CHECK-LABEL: func.func @local_rows
// CHECK-SAME: (%[[PID:.*]]: index)
func.func @local_rows(%pid: index) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: affine.for %[[L:.*]] = 0 to 16
  %r = affine.for %local = 0 to 16 iter_args(%acc = %c0) -> index {
    %g = affine.apply affine_map<(d0, d1) -> (d0 * 128 + d1)>(%pid, %local)
    %s = arith.addi %acc, %g : index
    affine.yield %s : index
  }
  return %r : index
}
