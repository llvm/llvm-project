// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s


// CHECK-LABEL: func @merge_poison()
//       CHECK:   %[[RES:.*]] = ub.poison : i32
//       CHECK:   return %[[RES]], %[[RES]]
func.func @merge_poison() -> (i32, i32) {
  %0 = ub.poison : i32
  %1 = ub.poison : i32
  return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: func @drop_ops_before_unreachable( 
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   vector.print
//  CHECK-NEXT:   scf.for {{.*}} {
//  CHECK-NEXT:     vector.print
//  CHECK-NEXT:   }
//  CHECK-NEXT:   ub.unreachable
func.func @drop_ops_before_unreachable(%arg0: i32) {
  %lb = arith.constant 3 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 0 : index
  vector.print %arg0 : i32
  // Infinite loop that may not progress. Such ops (and everything before)
  // is not erased.
  scf.for %iv = %lb to %ub step %step {
    vector.print %arg0 : i32
  } {mustProgress = false}
  vector.print %arg0 : i32
  ub.unreachable
}
