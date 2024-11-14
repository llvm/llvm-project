// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @addi_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@addi_i1\n"
  %res = arith.addi %v1, %v2 : i1
  vector.print %res : i1
  return
}

func.func @addi() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------

  // addi on i1
  // addi(0, 1) : i1 = 1 : i1; addi(0, -1) : i1 = 1
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1

  // CHECK-LABEL: @addi_i1
  // CHECK-NEXT:  1
  func.call @addi_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @addi_i1
  // CHECK-NEXT:  1
  %true_based_on_non_zero_val = arith.constant -1 : i1
  func.call @addi_i1(%false, %true_based_on_non_zero_val) : (i1, i1) -> ()

  // ------------------------------------------------
  // TODO: Test i8, i16 etc..
  // ------------------------------------------------

  return
}

func.func @addui_extended_i1(%v1 : i1, %v2 : i1) {
  vector.print str "@addui_extended_i1\n"
  %res, %overflow = arith.addui_extended %v1, %v2 : i1, i1
  vector.print %res : i1
  vector.print %overflow : i1
  return
}

func.func @addi_extended() {
  // ------------------------------------------------
  // Test i1
  // ------------------------------------------------

  // addui_extended on i1
  // addui_extended 1 1 : i1 = 0, 1
  %true = arith.constant 1 : i1
  %false = arith.constant 0 : i1
  
  // CHECK-LABEL: @addui_extended_i1
  // CHECK-NEXT:  0
  // CHECK-NEXT:  1
  func.call @addui_extended_i1(%true, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @addui_extended_i1
  // CHECK-NEXT:  1
  // CHECK-NEXT:  0
  func.call @addui_extended_i1(%true, %false) : (i1, i1) -> ()

  // CHECK-LABEL: @addui_extended_i1
  // CHECK-NEXT:  1
  // CHECK-NEXT:  0
  func.call @addui_extended_i1(%false, %true) : (i1, i1) -> ()

  // CHECK-LABEL: @addui_extended_i1
  // CHECK-NEXT:  0
  // CHECK-NEXT:  0
  func.call @addui_extended_i1(%false, %false) : (i1, i1) -> ()

  // ------------------------------------------------
  // TODO: Test i8, i16 etc.. 
  // ------------------------------------------------
  return
}

func.func @entry() {
  func.call @addi() : () -> ()
  func.call @addi_extended() : () -> ()
  return
}
