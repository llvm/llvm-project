// Tests arith operations on i1 type.
// These tests are intended to be target agnostic: they should yield the same results 
// regardless of the target platform.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @add_i1(%v1 : i1, %v2 : i1) -> (i1) {
  %res = arith.addi %v1, %v2 : i1
  return %res : i1
}

func.func @additions_i1() {
  // addi on i1
  // addi(0, 1) : i1 = 1 : i1; addi(0, -1) : i1 = 1
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1

  // CHECK:      1
  %0 = func.call @add_i1(%false, %true) : (i1, i1) -> i1
  vector.print %0 : i1

  // CHECK-NEXT: 1
  %true_based_on_non_zero_val = arith.constant -1 : i1
  %1 = func.call @add_i1(%false, %true_based_on_non_zero_val) : (i1, i1) -> i1
  vector.print %1 : i1
  return
}

func.func @addui_extended_i1(%v1 : i1, %v2 : i1) -> (i1, i1) {
  %res, %overflow = arith.addui_extended %v1, %v2 : i1, i1
  return %res, %overflow : i1, i1
}

func.func @additions_extended_i1() {
  // addui_extended on i1
  // addui_extended 1 1 : i1 = 0, 1
  %true = arith.constant 1 : i1
  %false = arith.constant 0 : i1
  
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  %sum_tt, %overflow_tt = func.call @addui_extended_i1(%true, %true) : (i1, i1) -> (i1, i1)
  vector.print %sum_tt : i1
  vector.print %overflow_tt : i1

  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  %sum_tf, %overflow_tf = func.call @addui_extended_i1(%true, %false) : (i1, i1) -> (i1, i1)
  vector.print %sum_tf : i1
  vector.print %overflow_tf : i1

  // CHECK-NEXT: 1
  // CHECK-NEXT: 0
  %sum_ft, %overflow_ft = func.call @addui_extended_i1(%false, %true) : (i1, i1) -> (i1, i1)
  vector.print %sum_ft : i1
  vector.print %overflow_ft : i1

  // CHECK-NEXT: 0
  // CHECK-NEXT: 0
  %sum_ff, %overflow_ff = func.call @addui_extended_i1(%false, %false) : (i1, i1) -> (i1, i1)
  vector.print %sum_ff : i1
  vector.print %overflow_ff : i1
  return
}

func.func @addui_extended_overflow_bit_is_treated_as_n1_in_comparisons() {
  // check that addui_extended overflow bit is treated as -1 in comparison operations
  //  in the case of an overflow
  // addui_extended -1 -1 = (..., overflow_bit) 
  // assert(overflow_bit <= 0)
  %1 = arith.constant -1 : i64
  %sum, %overflow = arith.addui_extended %1, %1 : i64, i1
  %false = arith.constant false
  %0 = arith.cmpi sge, %overflow, %false : i1

  // CHECK-NEXT: 0
  vector.print %0 : i1
  return
}

func.func @entry() {
  func.call @additions_i1() : () -> ()
  func.call @additions_extended_i1() : () -> ()
  func.call @addui_extended_overflow_bit_is_treated_as_n1_in_comparisons() : () -> ()
  return
}
