// tests arith operations on i1 type.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @mulsi_extended_on_i1() {
    // mulsi_extended on i1, tests for overflow bit
    // mulsi_extended 1, 1 : i1 = (1, 0)
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    %true = arith.constant true
    %low, %high = arith.mulsi_extended %true, %true : i1
    vector.print %low : i1
    vector.print %high : i1
    return
}

func.func @mulsi_mului_extended_overflows() {
    // mulsi and mului extended versions, with overflow
    // mulsi_extended -100, -100 : i8 = (16, 39); mului_extended -100, -100 : i8 = (16, 95)
    // CHECK-NEXT:  16
    // CHECK-NEXT:  39
    // CHECK-NEXT:  16
    // CHECK-NEXT:  95
    %c-100_i8 = arith.constant -100 : i8
    %low, %high = arith.mulsi_extended %c-100_i8, %c-100_i8 : i8
    vector.print %low : i8
    vector.print %high : i8
    %low_0, %high_1 = arith.mului_extended %c-100_i8, %c-100_i8 : i8
    vector.print %low_0 : i8
    vector.print %high_1 : i8
    return
}

func.func @entry() {
    func.call @mulsi_extended_on_i1() : () -> ()
    func.call @mulsi_mului_extended_overflows() : () -> ()
    return
}
