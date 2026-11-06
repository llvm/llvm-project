// RUN: mlir-opt %s --test-lower-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @shrsi_i8(%v1 : i8, %v2 : i8) {
  vector.print str "@shrsi_i8\n"
  %res = arith.shrsi %v1, %v2 : i8
  vector.print %res : i8
  return
}

func.func @shrui_i8(%v1 : i8, %v2 : i8) {
  vector.print str "@shrui_i8\n"
  %res = arith.shrui %v1, %v2 : i8
  vector.print %res : i8
  return
}

func.func @shli_i8(%v1 : i8, %v2 : i8) {
  vector.print str "@shli_i8\n"
  %res = arith.shli %v1, %v2 : i8
  vector.print %res : i8
  return
}

func.func @shrsi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@shrsi_i1\n"
  %res = arith.shrsi %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @shrui_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@shrui_i1\n"
  %res = arith.shrui %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @shli_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@shli_i1\n"
  %res = arith.shli %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @shrsi() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------
  %false = arith.constant 0 : i1

  // shift by zero : i1 should be non poison
  // shrsi 0 0 : i1 = 0
  // CHECK-LABEL: @shrsi_i1
  // CHECK-NEXT:  0
  func.call @shrsi_i1(%false, %false) : (i1, i1) -> ()

  // ------------------------------------------------
  // Test i8
  // ------------------------------------------------
  %c7 = arith.constant 7 : i8
  %cn10 = arith.constant -10 : i8
  %c0 = arith.constant 0 : i8

  // shrsi preserves signs
  // shrsi -10 7 : i8 = -1
  // CHECK-LABEL: @shrsi_i8
  // CHECK-NEXT:  -1
  func.call @shrsi_i8(%cn10, %c7) : (i8, i8) -> ()

  // shift on zero is identity
  // shrsi 7 0 : i8 = 7
  // CHECK-LABEL: @shrsi_i8
  // CHECK-NEXT:  7
  func.call @shrsi_i8(%c7, %c0) : (i8, i8) -> ()

  // ------------------------------------------------
  // TODO: Test i16, i32 etc..
  // ------------------------------------------------

  return
}

func.func @shrui() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------
  %false = arith.constant 0 : i1

  // shift by zero : i1 should be non poison
  // shrui 0 0 : i1 = 0
  // CHECK-LABEL: @shrui_i1
  // CHECK-NEXT:  0
  func.call @shrui_i1(%false, %false) : (i1, i1) -> ()

  // ------------------------------------------------
  // Test i8
  // ------------------------------------------------
  %cn10 = arith.constant -10 : i8
  %c0 = arith.constant 0 : i8

  // shift on zero is identity
  // shrsi -10 0 : i8 = -10
  // CHECK-LABEL: @shrui_i8
  // CHECK-NEXT:  -10
  func.call @shrui_i8(%cn10, %c0) : (i8, i8) -> ()

  // ------------------------------------------------
  // TODO: Test i16, i32 etc..
  // ------------------------------------------------

  return
}

func.func @shli() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------
  %false = arith.constant 0 : i1

  // shift by zero : i1 should be non poison
  // shli 0 0 : i1 = 0
  // CHECK-LABEL: @shli_i1
  // CHECK-NEXT:  0
  func.call @shli_i1(%false, %false) : (i1, i1) -> ()

  // ------------------------------------------------
  // Test i8
  // ------------------------------------------------
  %c7 = arith.constant 7 : i8
  %c0 = arith.constant 0 : i8
  %cn100 = arith.constant -100 : i8

  // shift on zero is identity
  // shli 7 0 : i8 = 7
  // CHECK-LABEL: @shli_i8
  // CHECK-NEXT:  7
  func.call @shli_i8(%c7, %c0) : (i8, i8) -> ()

  // shli on i8, value goes off into the void (overflow/modulus needed)
  // shli (-100), 7
  // CHECK-LABEL: @shli_i8
  // CHECK-NEXT:  0
  func.call @shli_i8(%cn100, %c7) : (i8, i8) -> ()

  // ------------------------------------------------
  // TODO: Test i16, i32 etc..
  // ------------------------------------------------

  return
}

func.func @entry() {
  func.call @shrsi() : () -> ()
  func.call @shrui() : () -> ()
  func.call @shli() : () -> ()
  return
}
