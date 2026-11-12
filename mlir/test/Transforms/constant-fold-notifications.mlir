// RUN: mlir-opt %s -test-single-fold="dump-notifications" | FileCheck %s

// Folding the comparisons and their operands reuses the constant 2. Moving
// that constant before the earlier definition must notify the listener.
// CHECK: notifyOperationInserted: arith.constant, previous = arith.constant
// CHECK-NEXT: notifyOperationErased: arith.constant
// CHECK: func.func @fold_and_reuse()
// CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
// CHECK-NEXT: %[[FALSE:.*]] = arith.constant false
// CHECK-NEXT: return %[[TRUE]], %[[FALSE]]
func.func @fold_and_reuse() -> (i1, i1) {
  %a = arith.constant 4 : i32
  %b = arith.constant 2 : i32
  %c = arith.constant 1 : i32
  %d = arith.constant 2 : i32
  %true = arith.cmpi sgt, %a, %b : i32
  %false = arith.cmpi sgt, %c, %d : i32
  return %true, %false : i1, i1
}
