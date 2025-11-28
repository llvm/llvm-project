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

// CHECK-LABEL: func @drop_ops_before_unreachable() 
//  CHECK-NEXT:   ub.unreachable
func.func @drop_ops_before_unreachable() {
  "test.foo"() : () -> ()
  "test.bar"() : () -> ()
  ub.unreachable
}
