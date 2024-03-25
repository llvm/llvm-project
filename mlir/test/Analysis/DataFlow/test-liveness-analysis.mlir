// RUN: mlir-opt -split-input-file -test-liveness-analysis %s 2>&1 | FileCheck %s

// Positive test: Type (1.a) "is an operand of an op with memory effects"
// zero is live because it is stored in memory.
// CHECK-LABEL: test_tag: zero:
// CHECK-NEXT:  result #0: live
func.func @test_1_type_1.a(%arg0: memref<i32>) {
  %c0_i32 = arith.constant {tag = "zero"} 0 : i32
  memref.store %c0_i32, %arg0[] : memref<i32>
  return
}

// -----

// Positive test: Type (1.b) "is a non-forwarded branch operand and a block
// where its op could take the control has an op with memory effects"
// %arg2 is live because it can make the control go into a block with a memory
// effecting op.
// CHECK-LABEL: test_tag: br:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  operand #2: live
func.func @test_2_RegionBranchOpInterface_type_1.b(%arg0: memref<i32>, %arg1: memref<i32>, %arg2: i1) {
  %c0_i32 = arith.constant 0 : i32
  cf.cond_br %arg2, ^bb1(%c0_i32 : i32), ^bb2(%c0_i32 : i32) {tag = "br"}
^bb1(%0 : i32):
  memref.store %0, %arg0[] : memref<i32>
  cf.br ^bb3
^bb2(%1 : i32):
  memref.store %1, %arg1[] : memref<i32>
  cf.br ^bb3
^bb3:
  return
}

// -----

// Positive test: Type (1.b) "is a non-forwarded branch operand and a block
// where its op could take the control has an op with memory effects"
// %arg0 is live because it can make the control go into a block with a memory
// effecting op.
// CHECK-LABEL: test_tag: flag:
// CHECK-NEXT:  operand #0: live
func.func @test_3_BranchOpInterface_type_1.b(%arg0: i32, %arg1: memref<i32>, %arg2: memref<i32>) {
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

// -----

func.func private @private(%arg0 : i32, %arg1 : i32) {
  func.return
}

// Positive test: Type (1.c) "is a non-forwarded call operand"
// CHECK-LABEL: test_tag: call
// CHECK-LABEL:  operand #0: not live
// CHECK-LABEL:  operand #1: not live
// CHECK-LABEL:  operand #2: live
func.func @test_4_type_1.c(%arg0: i32, %arg1: i32, %device: i32, %m0: memref<i32>) {
  test.call_on_device @private(%arg0, %arg1), %device {tag = "call"} : (i32, i32, i32) -> ()
  return
}

// -----

// Positive test: Type (2) "is returned by a public function"
// zero is live because it is returned by a public function.
// CHECK-LABEL: test_tag: zero:
// CHECK-NEXT:  result #0: live
func.func @test_5_type_2() -> (f32){
  %0 = arith.constant {tag = "zero"} 0.0 : f32
  return %0 : f32
}

// -----

// Positive test: Type (3) "is used to compute a value of type (1) or (2)"
// %arg1 is live because the scf.while has a live result and %arg1 is a
// non-forwarded branch operand.
// %arg2 is live because it is forwarded to the live result of the scf.while
// op.
// %arg5 is live because it is forwarded to %arg8 which is live.
// %arg8 is live because it is forwarded to %arg4 which is live as it writes
// to memory.
// Negative test:
// %arg3 is not live even though %arg1, %arg2, and %arg5 are live because it
// is neither a non-forwarded branch operand nor a forwarded operand that
// forwards to a live value. It actually is a forwarded operand that forwards
// to non-live values %0#1 and %arg7.
// CHECK-LABEL: test_tag: condition:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  operand #2: not live
// CHECK-NEXT:  operand #3: live
// CHECK-LABEL: test_tag: add:
// CHECK-NEXT:  operand #0: live
func.func @test_6_RegionBranchTerminatorOpInterface_type_3(%arg0: memref<i32>, %arg1: i1) -> (i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0:3 = scf.while (%arg2 = %c0_i32, %arg3 = %c1_i32, %arg4 = %c2_i32, %arg5 = %c2_i32) : (i32, i32, i32, i32) -> (i32, i32, i32) {
    memref.store %arg4, %arg0[] : memref<i32>
    scf.condition(%arg1) {tag = "condition"} %arg2, %arg3, %arg5 : i32, i32, i32
  } do {
  ^bb0(%arg6: i32, %arg7: i32, %arg8: i32):
    %1 = arith.addi %arg8, %arg8 {tag = "add"} : i32
    %c3_i32 = arith.constant 3 : i32
    scf.yield %arg6, %arg7, %arg8, %c3_i32 : i32, i32, i32, i32
  }
  return %0#0 : i32
}

// -----

func.func private @private0(%0 : i32) -> i32 {
  %1 = arith.addi %0, %0 {tag = "in_private0"} : i32
  func.return %1 : i32
}

// Positive test: Type (3) "is used to compute a value of type (1) or (2)"
// zero, ten, and one are live because they are used to decide the number of
// times the `for` loop executes, which in turn decides the value stored in
// memory.
// in_private0 and x are also live because they decide the value stored in
// memory.
// Negative test:
// y is not live even though the non-forwarded branch operand and x are live.
// CHECK-LABEL: test_tag: in_private0:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: zero:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: ten:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: one:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: x:
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: y:
// CHECK-NEXT:  result #0: not live
func.func @test_7_type_3(%arg0: memref<i32>) {
  %c0 = arith.constant {tag = "zero"} 0 : index
  %c10 = arith.constant {tag = "ten"} 10 : index
  %c1 = arith.constant {tag = "one"} 1 : index
  %x = arith.constant {tag = "x"} 0 : i32
  %y = arith.constant {tag = "y"} 1 : i32
  %0:2 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %x, %arg3 = %y) -> (i32, i32) {
    %1 = arith.addi %x, %x : i32
    %2 = func.call @private0(%1) : (i32) -> i32
    scf.yield %2, %arg3 : i32, i32
  }
  memref.store %0#0, %arg0[] : memref<i32>
  return
}

// -----

func.func private @private1(%0 : i32) -> i32 {
  %1 = func.call @private2(%0) : (i32) -> i32
  %2 = arith.muli %0, %1 {tag = "in_private1"} : i32
  func.return %2 : i32
}

func.func private @private2(%0 : i32) -> i32 {
  %cond = arith.index_cast %0 {tag = "in_private2"} : i32 to index
  %1 = scf.index_switch %cond -> i32
  case 1 {
    %ten = arith.constant 10 : i32
    scf.yield %ten : i32
  }
  case 2 {
    %twenty = arith.constant 20 : i32
    scf.yield %twenty : i32
  }
  default {
    %thirty = arith.constant 30 : i32
    scf.yield %thirty : i32
  }
  func.return %1 : i32
}

// Positive test: Type (3) "is used to compute a value of type (1) or (2)"
// in_private1, in_private2, and final are live because they are used to compute
// the value returned by this public function.
// CHECK-LABEL: test_tag: in_private1:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: in_private2:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  result #0: live
// CHECK-LABEL: test_tag: final:
// CHECK-NEXT:  operand #0: live
// CHECK-NEXT:  operand #1: live
// CHECK-NEXT:  result #0: live
func.func @test_8_type_3(%arg: i32) -> (i32) {
  %0 = func.call @private1(%arg) : (i32) -> i32
  %final = arith.muli %0, %arg {tag = "final"} : i32
  return %final : i32
}

// -----

// Negative test: None of the types (1), (2), or (3)
// zero is not live because it has no effect outside the program: it doesn't
// affect the memory or the program output.
// CHECK-LABEL: test_tag: zero:
// CHECK-NEXT:  result #0: not live
// CHECK-LABEL: test_tag: one:
// CHECK-NEXT:  result #0: live
func.func @test_9_negative() -> (f32){
  %0 = arith.constant {tag = "zero"} 0.0 : f32
  %1 = arith.constant {tag = "one"} 1.0 : f32
  return %1 : f32
}

// -----

// Negative test: None of the types (1), (2), or (3)
// %1 is not live because it has no effect outside the program: it doesn't
// affect the memory or the program output. Even though it is returned by the
// function `@private_1`, it is never used by the caller.
// Note that this test clearly shows how this liveness analysis utility differs
// from the existing liveness utility present at
// llvm-project/mlir/include/mlir/Analysis/Liveness.h. The latter marks %1 as
// live as it exists the block of function `@private_1`, simply because it is
// computed inside and returned by the block, irrespective of whether or not it
// is used by the caller.
// CHECK-LABEL: test_tag: one:
// CHECK:  result #0: not live
func.func private @private_1() -> (i32, i32) {
  %0 = arith.constant 0 : i32
  %1 = arith.addi %0, %0 {tag = "one"} : i32
  return %0, %1 : i32, i32
}
func.func @test_10_negative() -> (i32) {
  %0:2 = func.call @private_1() : () -> (i32, i32)
  return %0#0 : i32
}
