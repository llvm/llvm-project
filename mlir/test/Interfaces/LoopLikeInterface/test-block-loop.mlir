// RUN: mlir-opt %s --mlir-disable-threading -test-block-is-in-loop 2>&1 | FileCheck %s

module {
  // Test function with only one bb
  func.func @simple() {
    func.return
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0:

  // Test simple loop bb0 -> bb0
  func.func @loopForever() {
  ^bb0:
    cf.br ^bb1
  ^bb1:
    cf.br ^bb1
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb1:

  // Test bb0 -> bb1 -> bb2 -> bb1
  func.func @loopForever2() {
  ^bb0:
    cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb1
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb1:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb2:

  // Test conditional branch without loop
  // bb0 -> bb1 -> {bb2, bb3}
  func.func @noLoop(%arg0: i1) {
    cf.br ^bb1
  ^bb1:
    cf.cond_br %arg0, ^bb2, ^bb3
  ^bb2:
    func.return
  ^bb3:
    func.return
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0(%arg0: i1)
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb1:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb2:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb3:

  // test multiple loops
  // bb0 -> bb1 -> bb2 -> bb3 { -> bb2} -> bb4 { -> bb1 } -> bb5
  func.func @multipleLoops(%arg0: i1, %arg1: i1) {
    cf.br ^bb1
  ^bb1:
    cf.br ^bb2
  ^bb2:
    cf.br ^bb3
  ^bb3:
    cf.cond_br %arg0, ^bb4, ^bb2
  ^bb4:
    cf.cond_br %arg1, ^bb1, ^bb5
  ^bb5:
    return
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0(%arg0: i1, %arg1: i1)
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb1:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb2:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb3:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb4:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb5:

  // test derived from real Flang output
  func.func @_QPblockTest0(%arg0: i1, %arg1: i1) {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb4
    cf.cond_br %arg0, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    cf.cond_br %arg1, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    return
  ^bb4:  // pred: ^bb2
    cf.br ^bb1
  ^bb5:  // pred: ^bb1
    return
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0(%arg0: i1, %arg1: i1)
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb1:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb2:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb3:
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb4:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb5:

// check nested blocks
  func.func @check_alloc_in_loop(%counter : i64) {
    cf.br ^bb1(%counter: i64)
    ^bb1(%lv : i64):
      %cm1 = arith.constant -1 : i64
      %rem = arith.addi %lv, %cm1 : i64
      %zero = arith.constant 0 : i64
      %p = arith.cmpi eq, %rem, %zero : i64
      cf.cond_br %p, ^bb3, ^bb2
    ^bb2:
      scf.execute_region -> () {
        %c1 = arith.constant 1 : i64
        scf.yield
      }
      cf.br ^bb1(%rem: i64)
    ^bb3:
      return
  }
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb0(%arg0: i64):
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb1(%0: i64)
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb0:
// CHECK-NEXT: %c1_i64
// CHECK: Block is in a loop
// CHECK-NEXT: ^bb2:
// CHECK: Block is not in a loop
// CHECK-NEXT: ^bb3:
}
