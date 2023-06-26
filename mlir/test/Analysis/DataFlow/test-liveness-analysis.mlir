// RUN: mlir-opt -split-input-file -test-liveness-analysis %s 2>&1 | FileCheck %s

// This is live because it is stored in memory if the `if` block executes.
// CHECK-LABEL: test_tag: c0:
// CHECK-NEXT:  result #0: live

// This is not live because it is neither stored in memory nor used to compute
// such a value.
// CHECK-LABEL: test_tag: c1:
// CHECK-NEXT:  result #0: not live

// This is live because it is stored in memory if the `else` block executes.
// CHECK-LABEL: test_tag: c2:
// CHECK-NEXT:  result #0: live

// These are live because they are used to decide whether the `if` block executes
// or the `else` one, which in turn decides the value stored in memory.
// Note that if `visitBranchOperand()` was left empty, they would have been
// incorrectly marked as "not live".
// CHECK-LABEL: test_tag: condition0:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  result #0: live
module {
  func.func @test_simple_and_if(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i1) {
    %c0_i32 = arith.constant {tag = "c0"} 0 : i32
    %c1_i32 = arith.constant {tag = "c1"} 1 : i32
    %c2_i32 = arith.constant {tag = "c2"} 2 : i32
    %0 = arith.addi %arg2, %arg2 {tag = "condition0"} : i1
    %1 = scf.if %0 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      scf.yield %c2_i32 : i32
    }
    memref.store %1, %arg0[] : memref<i32>
    return
  }
}

// -----

// zero, ten, and one are live because they are used to decide the number of
// times the `for` loop executes, which in turn decides the value stored in
// memory.
// Note that if `visitBranchOperand()` was left empty, they would have been
// incorrectly marked as "not live".
// CHECK-LABEL: test_tag: zero:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: ten:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: one:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: x:
// CHECK-NEXT:  result #0: live
module {
  func.func @test_for(%arg0: memref<i32>) {
    %c0 = arith.constant {tag = "zero"} 0 : index
    %c10 = arith.constant {tag = "ten"} 10 : index
    %c1 = arith.constant {tag = "one"} 1 : index
    %x = arith.constant {tag = "x"} 0 : i32
    %0 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %x) -> (i32) {
      %1 = arith.addi %x, %x : i32
      scf.yield %1 : i32
    }
    memref.store %0, %arg0[] : memref<i32>
    return
  }
}

// -----

// This is live because it is used to decide which switch case executes, which
// in turn decides the value stored in memory.
// Note that if `visitBranchOperand()` was left empty, it would have been
// incorrectly marked as "not live".
// CHECK-LABEL: test_tag: switch:
// CHECK-NEXT:  operand #0: live
module {
  func.func @test_scf_switch(%arg0: index, %arg1: memref<i32>) {
    %0 = scf.index_switch %arg0 {tag = "switch"} -> i32
    case 1 {
      %c10_i32 = arith.constant 10 : i32
      scf.yield %c10_i32 : i32
    }
    case 2 {
      %c20_i32 = arith.constant 20 : i32
      scf.yield %c20_i32 : i32
    }
    default {
      %c30_i32 = arith.constant 30 : i32
      scf.yield %c30_i32 : i32
    }
    memref.store %0, %arg1[] : memref<i32>
    return
  }
}

// -----

// The branch operand is incorrectly marked "not live" because, for some reason
// unclear to me yet, it is not visited by the `visitBranchOperand()` function.
// CHECK-LABEL: test_tag: br:
// CHECK-NEXT:  operand #0: not live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  operand #2: live
module {
  func.func @test_blocks(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: memref<i32>, %arg3: i1) {
    %c0_i32 = arith.constant 0 : i32
    cf.cond_br %arg3, ^bb1(%c0_i32 : i32), ^bb2(%c0_i32 : i32) {tag = "br"}
  ^bb1(%0 : i32):
    memref.store %0, %arg0[] : memref<i32>
    cf.br ^bb3
  ^bb2(%1 : i32):
    memref.store %1, %arg1[] : memref<i32>
    cf.br ^bb3
  ^bb3:
    return
  }
}

// -----

// The branch operand is incorrectly marked "not live" because, for some reason
// unclear to me yet, it is not visited by the `visitBranchOperand()` function.
// CHECK-LABEL: test_tag: flag:
// CHECK-NEXT:  operand #0: not live
module {
  func.func @test_switch(%arg0: i32, %arg1: memref<i32>, %arg2: memref<i32>) {
    %c0_i32 = arith.constant 0 : i32
    cf.switch %arg0 : i32, [
      default: ^bb1,
      42: ^bb2
    ] {tag = "flag"}
  ^bb1:
    memref.store %c0_i32, %arg1[] : memref<i32>
    cf.br ^bb3
  ^bb2:
    memref.store %c0_i32, %arg2[] : memref<i32>
    cf.br ^bb3
  ^bb3:
    return
  }
}

// -----

// The branch operand is incorrectly marked "not live" because, for some reason
// unclear to me yet, it is not visited by the `visitBranchOperand()` function.
// CHECK-LABEL: test_tag: condition:
// CHECK-NEXT:  operand #0: not live
// CHECK-NEXT:  operand #1: live
module {
  func.func @test_condition(%arg0: memref<i32>, %arg1: i32, %arg2: i1) {
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.while (%arg3 = %c0_i32) : (i32) -> (i32) {
      memref.store %arg3, %arg0[] : memref<i32>
      scf.condition(%arg2) {tag = "condition"} %arg3 : i32
    } do {
    ^bb0(%arg3: i32):
      memref.store %arg3, %arg0[] : memref<i32>
      scf.yield %arg3 : i32
    }
    memref.store %0, %arg0[] : memref<i32>
    return
  }
}
