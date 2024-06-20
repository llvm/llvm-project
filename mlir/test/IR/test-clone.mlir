// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(func.func(test-clone))" | FileCheck %s

module {
  func.func @fixpoint(%arg1 : i32) -> i32 {
    %r = "test.use"(%arg1) ({
      %r2 = "test.use2"(%arg1) ({
         "test.yield2"(%arg1) : (i32) -> ()
      }) : (i32) -> i32
      "test.yield"(%r2) : (i32) -> ()
    }) : (i32) -> i32
    return %r : i32
  }
}

// CHECK: notifyOperationInserted: test.use
// CHECK-NEXT: notifyOperationInserted: test.use2
// CHECK-NEXT: notifyOperationInserted: test.yield2
// CHECK-NEXT: notifyOperationInserted: test.yield
// CHECK-NEXT: notifyOperationInserted: func.return

// CHECK:   func @fixpoint(%[[arg0:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[i0:.+]] = "test.use"(%[[arg0]]) ({
// CHECK-NEXT:       %[[r2:.+]] = "test.use2"(%[[arg0]]) ({
// CHECK-NEXT:         "test.yield2"(%[[arg0]]) : (i32) -> ()
// CHECK-NEXT:       }) : (i32) -> i32
// CHECK-NEXT:       "test.yield"(%[[r2]]) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     %[[i1:.+]] = "test.use"(%[[i0]]) ({
// CHECK-NEXT:       %[[r2:.+]] = "test.use2"(%[[i0]]) ({
// CHECK-NEXT:         "test.yield2"(%[[i0]]) : (i32) -> ()
// CHECK-NEXT:       }) : (i32) -> i32
// CHECK-NEXT:       "test.yield"(%[[r2]]) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     return %[[i1]] : i32
// CHECK-NEXT:   }
