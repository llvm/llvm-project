// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @mulsi_extended_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@mulsi_extended_i1\n"
  %low, %high = arith.mulsi_extended %v1, %v2 : i1
  vector.print %low : i1
  vector.print %high : i1
  return
}

func.func @mulsi_extended_i8(%v1 : i8, %v2 : i8) {
  vector.print str "@mulsi_extended_i8\n"
  %low, %high = arith.mulsi_extended %v1, %v2 : i8
  vector.print %low : i8
  vector.print %high : i8
  return
}

func.func @mulsi_extended() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------

  // mulsi_extended on i1, tests for overflow bit
  // mulsi_extended 1, 1 : i1 = (1, 0)
  %true = arith.constant true
  %false = arith.constant false

  // CHECK-LABEL: @mulsi_extended_i1
  // CHECK-NEXT:  1
  // CHECK-NEXT:  0
  func.call @mulsi_extended_i1(%true, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @mulsi_extended_i1
  // CHECK-NEXT:  0
  // CHECK-NEXT:  0
  func.call @mulsi_extended_i1(%true, %false) : (i1, i1) -> ()

  // CHECK-LABEL: @mulsi_extended_i1
  // CHECK-NEXT:  0
  // CHECK-NEXT:  0
  func.call @mulsi_extended_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @mulsi_extended_i1
  // CHECK-NEXT:  0
  // CHECK-NEXT:  0
  func.call @mulsi_extended_i1(%false, %false) : (i1, i1) -> ()

  // ------------------------------------------------
  // Test i8
  // ------------------------------------------------
  // mulsi extended versions, with overflow
  %c_100_i8 = arith.constant -100 : i8

  // mulsi_extended -100, -100 : i8 = (16, 39)
  // CHECK-LABEL: @mulsi_extended_i8
  // CHECK-NEXT:  16
  // CHECK-NEXT:  39
  func.call @mulsi_extended_i8(%c_100_i8, %c_100_i8) : (i8, i8) -> ()

  // ------------------------------------------------
  // TODO: Test i16, i32 etc.. 
  // ------------------------------------------------
  return
}

func.func @mului_extended_i8(%v1 : i8, %v2 : i8) {
  vector.print str "@mului_extended_i8\n"
  %low, %high = arith.mului_extended %v1, %v2 : i8
  vector.print %low : i8
  vector.print %high : i8
  return
}

func.func @mului_extended() {
  // ------------------------------------------------
  // Test i8
  // ------------------------------------------------
  %c_n100_i8 = arith.constant -100 : i8
  %c_156_i8 = arith.constant 156 : i8

  // mului_extended -100, -100 : i8 = (16, 95)
  // and on equivalent representations (e.g. 156 === -100 (mod 256))

  // CHECK-LABEL: @mului_extended_i8
  // CHECK-NEXT:  16
  // CHECK-NEXT:  95
  func.call @mului_extended_i8(%c_n100_i8, %c_n100_i8) : (i8, i8) -> ()

  // CHECK-LABEL: @mului_extended_i8
  // CHECK-NEXT:  16
  // CHECK-NEXT:  95
  func.call @mului_extended_i8(%c_n100_i8, %c_156_i8) : (i8, i8) -> ()

  // CHECK-LABEL: @mului_extended_i8
  // CHECK-NEXT:  16
  // CHECK-NEXT:  95
  func.call @mului_extended_i8(%c_156_i8, %c_n100_i8) : (i8, i8) -> ()

  // CHECK-LABEL: @mului_extended_i8
  // CHECK-NEXT:  16
  // CHECK-NEXT:  95
  func.call @mului_extended_i8(%c_156_i8, %c_156_i8) : (i8, i8) -> ()

  // ------------------------------------------------
  // TODO: Test i1, i16, i32 etc.. 
  // ------------------------------------------------
  return
}

func.func @entry() {
  func.call @mulsi_extended() : () -> ()
  func.call @mului_extended() : () -> ()
  return
}
