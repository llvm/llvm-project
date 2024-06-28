// Tests arith operations on i1 type.
// These tests are intended to be target agnostic: they should yield the same results 
// regardless of the target platform.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @zero_plus_one_on_i1() {
  // addi on i1
  // addi(0, 1) : i1 = 1 : i1; addi(0, -1) : i1 = 1
  // CHECK:      1
  // CHECK-NEXT: 1
  // CHECK-NEXT: 1
  %false = arith.constant 0 : i1
  %true = arith.constant 1 : i1
  %true_0 = arith.constant -1 : i1
  vector.print %true_0 : i1
  %0 = arith.addi %false, %true : i1
  vector.print %0 : i1
  %1 = arith.addi %false, %true_0 : i1
  vector.print %1 : i1
  return
}

func.func @addui_extended_i1() {
  // addui_extended on i1
  // addui_extended 1 1 : i1 = 0, 1
  // CHECK-NEXT: 0
  // CHECK-NEXT: 1
  %true = arith.constant 1 : i1
  %sum, %overflow = arith.addui_extended %true, %true : i1, i1
  vector.print %sum : i1
  vector.print %overflow : i1
  return
}

func.func @addui_extended_overflow_bit_is_n1() {
  // addui_extended overflow bit is treated as -1
  // addui_extended -1633386 -1643386 = ... 1 (overflow because negative numbers are large positive numbers)
  // CHECK-NEXT: 0
  %c-16433886_i64 = arith.constant -16433886 : i64
  %sum, %overflow = arith.addui_extended %c-16433886_i64, %c-16433886_i64 : i64, i1
  %false = arith.constant false
  %0 = arith.cmpi sge, %overflow, %false : i1
  vector.print %0 : i1 // but prints as "1"
  return
}

func.func @entry() {
  func.call @zero_plus_one_on_i1() : () -> ()
  func.call @addui_extended_i1() : () -> ()
  func.call @addui_extended_overflow_bit_is_n1() : () -> ()
  return
}
