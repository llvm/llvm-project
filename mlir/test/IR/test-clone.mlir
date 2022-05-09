// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="func.func(test-clone)" -split-input-file 

module {
  func.func @fixpoint(%arg1 : i32) -> i32 {
    %r = "test.use"(%arg1) ({
       "test.yield"(%arg1) : (i32) -> ()
    }) : (i32) -> i32
    return %r : i32
  }
}

// CHECK:   func @fixpoint(%[[arg0:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[i0:.+]] = "test.use"(%[[arg0]]) ({
// CHECK-NEXT:       "test.yield"(%arg0) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     %[[i1:.+]] = "test.use"(%[[i0]]) ({
// CHECK-NEXT:       "test.yield"(%[[i0]]) : (i32) -> ()
// CHECK-NEXT:     }) : (i32) -> i32
// CHECK-NEXT:     return %[[i1]] : i32
// CHECK-NEXT:   }
