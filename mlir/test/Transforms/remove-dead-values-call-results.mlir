// RUN: mlir-opt %s -remove-dead-values="canonicalize=0" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -remove-dead-values="canonicalize=1" -split-input-file | FileCheck %s

// The condition of a surviving scf.if must remain valid when the result of a
// recursive call in an unreachable function is removed.
// CHECK-LABEL: func.func private @dead() {
// CHECK-NEXT:    %[[COND:.*]] = ub.poison : i1
// CHECK-NEXT:    call @dead() : () -> ()
// CHECK-NEXT:    scf.if %[[COND]] {
// CHECK-NEXT:      call @dead() : () -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @dead() -> i1 {
  %cond = func.call @dead() : () -> i1
  scf.if %cond {
    %unused = func.call @dead() : () -> i1
  }
  %false = arith.constant false
  return %false : i1
}

// -----

// Keep the result that is live at another call site. Only the removed result
// with remaining uses needs a poison value, not the unused index result.
// CHECK-LABEL: func.func private @callee() -> i32 {
// CHECK-NEXT:    %[[C42:.*]] = arith.constant 42 : i32
// CHECK-NEXT:    return %[[C42]] : i32
// CHECK-NEXT:  }
func.func private @callee() -> (i32, i1, index) {
  %c42 = arith.constant 42 : i32
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  return %c42, %false, %c0 : i32, i1, index
}

// CHECK-LABEL: func.func private @unreachable_caller() {
// CHECK-NEXT:    %[[COND:.*]] = ub.poison : i1
// CHECK-NEXT:    %{{.*}} = call @callee() : () -> i32
// CHECK-NEXT:    scf.if %[[COND]] {
// CHECK-NEXT:      %{{.*}} = func.call @callee() : () -> i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @unreachable_caller() {
  %live, %cond, %unused = func.call @callee() : () -> (i32, i1, index)
  scf.if %cond {
    %unused_results:3 = func.call @callee() : () -> (i32, i1, index)
  }
  return
}

// CHECK-LABEL: func.func @live_caller() -> i32 {
// CHECK-NEXT:    %[[LIVE:.*]] = call @callee() : () -> i32
// CHECK-NEXT:    return %[[LIVE]] : i32
// CHECK-NEXT:  }
func.func @live_caller() -> i32 {
  %live, %unused_cond, %unused_index = func.call @callee() : () -> (i32, i1, index)
  return %live : i32
}
