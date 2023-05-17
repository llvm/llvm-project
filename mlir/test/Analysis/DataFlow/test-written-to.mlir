// RUN: mlir-opt -split-input-file -test-written-to %s 2>&1 | FileCheck %s

// CHECK-LABEL: test_tag: constant0
// CHECK: result #0: [a]
// CHECK-LABEL: test_tag: constant1
// CHECK: result #0: [b]
func.func @test_two_writes(%m0: memref<i32>, %m1: memref<i32>) -> (memref<i32>, memref<i32>) {
  %c0 = arith.constant {tag = "constant0"} 0 : i32
  %c1 = arith.constant {tag = "constant1"} 1 : i32
  memref.store %c0, %m0[] {tag_name = "a"} : memref<i32>
  memref.store %c1, %m1[] {tag_name = "b"} : memref<i32>
  return %m0, %m1 : memref<i32>, memref<i32>
}

// -----

// CHECK-LABEL: test_tag: c0
// CHECK: result #0: [b]
// CHECK-LABEL: test_tag: c1
// CHECK: result #0: [b]
// CHECK-LABEL: test_tag: condition
// CHECK: result #0: [brancharg0]
// CHECK-LABEL: test_tag: c2
// CHECK: result #0: [a]
// CHECK-LABEL: test_tag: c3
// CHECK: result #0: [a]
func.func @test_if(%m0: memref<i32>, %m1: memref<i32>, %condition: i1) {
  %c0 = arith.constant {tag = "c0"} 2 : i32
  %c1 = arith.constant {tag = "c1"} 3 : i32
  %condition2 = arith.addi %condition, %condition {tag = "condition"} : i1
  %0, %1 = scf.if %condition2 -> (i32, i32) {
    %c2 = arith.constant {tag = "c2"} 0 : i32
    scf.yield %c2, %c0: i32, i32
  } else {
    %c3 = arith.constant {tag = "c3"} 1 : i32
    scf.yield %c3, %c1: i32, i32
  }
  memref.store %0, %m0[] {tag_name = "a"} : memref<i32>
  memref.store %1, %m1[] {tag_name = "b"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: test_tag: c0
// CHECK: result #0: [a c]
// CHECK-LABEL: test_tag: c1
// CHECK: result #0: [b c]
// CHECK-LABEL: test_tag: br
// CHECK: operand #0: [brancharg0]
func.func @test_blocks(%m0: memref<i32>,
                       %m1: memref<i32>,
                       %m2: memref<i32>, %cond : i1) {
  %0 = arith.constant {tag = "c0"} 0 : i32
  %1 = arith.constant {tag = "c1"} 1 : i32
  cf.cond_br %cond, ^a(%0: i32), ^b(%1: i32) {tag = "br"}
^a(%a0: i32):
  memref.store %a0, %m0[] {tag_name = "a"} : memref<i32>
  cf.br ^c(%a0 : i32)
^b(%b0: i32):
  memref.store %b0, %m1[] {tag_name = "b"} : memref<i32>
  cf.br ^c(%b0 : i32)
^c(%c0 : i32):
  memref.store %c0, %m2[] {tag_name = "c"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: test_tag: two
// CHECK: result #0: [a]
func.func @test_infinite_loop(%m0: memref<i32>) {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.constant {tag = "two"} 2 : i32
  %3 = arith.constant -1 : i32
  cf.br ^loop(%0, %1, %2: i32, i32, i32)
^loop(%a: i32, %b: i32, %c: i32):
  memref.store %a, %m0[] {tag_name = "a"} : memref<i32>
  cf.br ^loop(%b, %c, %3 : i32, i32, i32)
}

// -----

// CHECK-LABEL: test_tag: c0
// CHECK: result #0: [a b c]
func.func @test_switch(%flag: i32, %m0: memref<i32>) {
  %0 = arith.constant {tag = "c0"} 0 : i32
  cf.switch %flag : i32, [
      default: ^a(%0 : i32),
      42: ^b(%0 : i32),
      43: ^c(%0 : i32)
  ]
^a(%a0: i32):
  memref.store %a0, %m0[] {tag_name = "a"} : memref<i32>
  cf.br ^c(%a0 : i32)
^b(%b0: i32):
  memref.store %b0, %m0[] {tag_name = "b"} : memref<i32>
  cf.br ^c(%b0 : i32)
^c(%c0 : i32):
  memref.store %c0, %m0[] {tag_name = "c"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: test_tag: add
// CHECK: result #0: [a]
func.func @test_caller(%m0: memref<f32>, %arg: f32) {
  %0 = arith.addf %arg, %arg {tag = "add"} : f32
  %1 = func.call @callee(%0) : (f32) -> f32
  %2 = arith.mulf %1, %1 : f32
  %3 = arith.mulf %2, %2 : f32
  %4 = arith.mulf %3, %3 : f32
  memref.store %4, %m0[] {tag_name = "a"} : memref<f32>
  return
}

func.func private @callee(%0 : f32) -> f32 {
  %1 = arith.mulf %0, %0 : f32
  %2 = arith.mulf %1, %1 : f32
  func.return %2 : f32
}

// -----

func.func private @callee(%0 : f32) -> f32 {
  %1 = arith.mulf %0, %0 : f32
  func.return %1 : f32
}

// CHECK-LABEL: test_tag: sub
// CHECK: result #0: [a]
func.func @test_caller_below_callee(%m0: memref<f32>, %arg: f32) {
  %0 = arith.subf %arg, %arg {tag = "sub"} : f32
  %1 = func.call @callee(%0) : (f32) -> f32
  memref.store %1, %m0[] {tag_name = "a"} : memref<f32>
  return
}

// -----

func.func private @callee1(%0 : f32) -> f32 {
  %1 = func.call @callee2(%0) : (f32) -> f32
  func.return %1 : f32
}

func.func private @callee2(%0 : f32) -> f32 {
  %1 = func.call @callee3(%0) : (f32) -> f32
  func.return %1 : f32
}

func.func private @callee3(%0 : f32) -> f32 {
  func.return %0 : f32
}

// CHECK-LABEL: test_tag: mul
// CHECK: result #0: [a]
func.func @test_callchain(%m0: memref<f32>, %arg: f32) {
  %0 = arith.mulf %arg, %arg {tag = "mul"} : f32
  %1 = func.call @callee1(%0) : (f32) -> f32
  memref.store %1, %m0[] {tag_name = "a"} : memref<f32>
  return
}

// -----

// CHECK-LABEL: test_tag: zero
// CHECK: result #0: [c]
// CHECK-LABEL: test_tag: init
// CHECK: result #0: [a b]
// CHECK-LABEL: test_tag: condition
// CHECK: operand #0: [brancharg0]
func.func @test_while(%m0: memref<i32>, %init : i32, %cond: i1) {
  %zero = arith.constant {tag = "zero"} 0 : i32
  %init2 = arith.addi %init, %init {tag = "init"} : i32
  %0, %1 = scf.while (%arg1 = %zero, %arg2 = %init2) : (i32, i32) -> (i32, i32) {
    memref.store %arg2, %m0[] {tag_name = "a"} : memref<i32>
    scf.condition(%cond) {tag = "condition"} %arg1, %arg2 : i32, i32
  } do {
   ^bb0(%arg1: i32, %arg2: i32):
    memref.store %arg1, %m0[] {tag_name = "c"} : memref<i32>
    %res = arith.addi %arg2, %arg2 : i32
    scf.yield %arg1, %res: i32, i32
  }
  memref.store %1, %m0[] {tag_name = "b"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: test_tag: zero
// CHECK: result #0: [brancharg0]
// CHECK-LABEL: test_tag: ten
// CHECK: result #0: [brancharg1]
// CHECK-LABEL: test_tag: one
// CHECK: result #0: [brancharg2]
// CHECK-LABEL: test_tag: x
// CHECK: result #0: [a]
func.func @test_for(%m0: memref<i32>) {
  %zero = arith.constant {tag = "zero"} 0 : index
  %ten = arith.constant {tag = "ten"} 10 : index
  %one = arith.constant {tag = "one"} 1 : index
  %x = arith.constant {tag = "x"} 0 : i32
  %0 = scf.for %i = %zero to %ten step %one iter_args(%ix = %x) -> (i32) {
    scf.yield %ix : i32
  }
  memref.store %0, %m0[] {tag_name = "a"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: test_tag: default_a
// CHECK-LABEL: result #0: [a]
// CHECK-LABEL: test_tag: default_b
// CHECK-LABEL: result #0: [b]
// CHECK-LABEL: test_tag: 1a
// CHECK-LABEL: result #0: [a]
// CHECK-LABEL: test_tag: 1b
// CHECK-LABEL: result #0: [b]
// CHECK-LABEL: test_tag: 2a
// CHECK-LABEL: result #0: [a]
// CHECK-LABEL: test_tag: 2b
// CHECK-LABEL: result #0: [b]
// CHECK-LABEL: test_tag: switch
// CHECK-LABEL: operand #0: [brancharg0]
func.func @test_switch(%arg0 : index, %m0: memref<i32>) {
  %0, %1 = scf.index_switch %arg0 {tag="switch"} -> i32, i32
  case 1 {
    %2 = arith.constant {tag="1a"} 10 : i32
    %3 = arith.constant {tag="1b"} 100 : i32
    scf.yield %2, %3 : i32, i32
  }
  case 2 {
    %4 = arith.constant {tag="2a"} 20 : i32
    %5 = arith.constant {tag="2b"} 200 : i32
    scf.yield %4, %5 : i32, i32
  }
  default {
    %6 = arith.constant {tag="default_a"} 30 : i32
    %7 = arith.constant {tag="default_b"} 300 : i32
    scf.yield %6, %7 : i32, i32
  }
  memref.store %0, %m0[] {tag_name = "a"} : memref<i32>
  memref.store %1, %m0[] {tag_name = "b"} : memref<i32>
  return
}

// -----

// CHECK-LABEL: llvm.func @decl(i64)
// CHECK-LABEL: llvm.func @func(%arg0: i64) {
// CHECK-NEXT:  llvm.call @decl(%arg0) : (i64) -> ()
// CHECK-NEXT:  llvm.return

llvm.func @decl(i64)

llvm.func @func(%lb : i64) -> () {
  llvm.call @decl(%lb) : (i64) -> ()
  llvm.return
} 
