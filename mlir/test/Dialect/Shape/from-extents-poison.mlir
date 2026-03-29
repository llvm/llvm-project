// RUN: mlir-opt %s -canonicalize="test-convergence" | FileCheck %s
//
// Regression tests for shape.from_extents crash with poison operands.
// Related GitHub issues:
//   - https://github.com/llvm/llvm-project/issues/179848
//   - https://github.com/llvm/llvm-project/issues/177951
//
// The crash occurred because FromExtentsOp::fold used cast<IntegerAttr>
// without checking the attribute kind. When ub.poison (PoisonAttr) was
// passed, the cast assertion failed. The fix uses dyn_cast_if_present
// and bails out early if any operand is not an IntegerAttr.

// -----

// GH#179848: Single poison extent should not crash and should not fold.
// CHECK-LABEL: func @from_extents_single_poison
func.func @from_extents_single_poison() -> !shape.shape {
  // CHECK: %[[POISON:.*]] = ub.poison : index
  // CHECK: %[[SHAPE:.*]] = shape.from_extents %[[POISON]] : index
  // CHECK: return %[[SHAPE]]
  %0 = ub.poison : index
  %ret = shape.from_extents %0 : index
  return %ret : !shape.shape
}

// -----

// GH#177951: Multiple poison extents should not crash.
// CHECK-LABEL: func @from_extents_multiple_poison
func.func @from_extents_multiple_poison() -> !shape.shape {
  // CHECK-DAG: %[[P1:.*]] = ub.poison : index
  // CHECK-DAG: %[[P2:.*]] = ub.poison : index
  // CHECK: shape.from_extents %[[P1]], %[[P2]]
  %p1 = ub.poison : index
  %p2 = ub.poison : index
  %ret = shape.from_extents %p1, %p2 : index, index
  return %ret : !shape.shape
}

// -----

// Mixed constant and poison should not fold (all operands must be IntegerAttr).
// CHECK-LABEL: func @from_extents_mixed_poison_constant
func.func @from_extents_mixed_poison_constant() -> !shape.shape {
  // CHECK-DAG: arith.constant 3 : index
  // CHECK-DAG: ub.poison : index
  // CHECK: shape.from_extents
  %c3 = arith.constant 3 : index
  %poison = ub.poison : index
  %ret = shape.from_extents %c3, %poison : index, index
  return %ret : !shape.shape
}

// -----

// Regression check: all-constant extents should still fold correctly.
// CHECK-LABEL: func @from_extents_all_constants_fold
func.func @from_extents_all_constants_fold() -> !shape.shape {
  // CHECK: shape.const_shape [2, 3, 4] : !shape.shape
  // CHECK-NOT: shape.from_extents
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %ret = shape.from_extents %c2, %c3, %c4 : index, index, index
  return %ret : !shape.shape
}

// -----

// Regression check: empty from_extents (rank-0) should still fold.
// CHECK-LABEL: func @from_extents_empty_fold
func.func @from_extents_empty_fold() -> !shape.shape {
  // CHECK: shape.const_shape [] : !shape.shape
  // CHECK-NOT: shape.from_extents
  %ret = shape.from_extents
  return %ret : !shape.shape
}

// -----

// Poison extent with downstream shape operations should not crash.
// CHECK-LABEL: func @from_extents_poison_with_rank
func.func @from_extents_poison_with_rank() -> index {
  // CHECK: ub.poison : index
  // CHECK: shape.from_extents
  // CHECK: shape.rank
  %poison = ub.poison : index
  %shape = shape.from_extents %poison : index
  %rank = shape.rank %shape : !shape.shape -> index
  return %rank : index
}

// -----

// Poison extent in broadcast should not crash.
// CHECK-LABEL: func @from_extents_poison_in_broadcast
func.func @from_extents_poison_in_broadcast(%arg0: !shape.shape) -> !shape.shape {
  // CHECK: ub.poison : index
  // CHECK: shape.from_extents
  // CHECK: shape.broadcast
  %poison = ub.poison : index
  %shape1 = shape.from_extents %poison : index
  %result = shape.broadcast %arg0, %shape1 : !shape.shape, !shape.shape -> !shape.shape
  return %result : !shape.shape
}

// -----

// Poison with shape.size type should also not crash.
// CHECK-LABEL: func @from_extents_poison_size_type
func.func @from_extents_poison_size_type() -> !shape.shape {
  // CHECK: ub.poison : !shape.size
  // CHECK: shape.from_extents
  %poison = ub.poison : !shape.size
  %ret = shape.from_extents %poison : !shape.size
  return %ret : !shape.shape
}
