// Tests arith operations on i1 type.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @signed_comparison_on_i1s() {
    // signed comparisons on i1s
    // slt 0 1 = false, sle 0 1 = false, sgt 0 1 = true, sge 0 1 = true
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    %false = arith.constant false
    %true = arith.constant true
    %0 = arith.cmpi slt, %false, %true : i1
    %1 = arith.cmpi sle, %false, %true : i1
    %2 = arith.cmpi sgt, %false, %true : i1
    %3 = arith.cmpi sge, %false, %true : i1
    vector.print %0 : i1
    vector.print %1 : i1
    vector.print %2 : i1
    vector.print %3 : i1
    return
}

func.func @sge_0_1_is_true() {
    // sge 0 -1, sge 0 1, should be true
    // sge 0 -1 == sge 0 1 == true (1)
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %true_0 = arith.constant -1 : i1
    %0 = arith.cmpi sge, %false, %true : i1
    %1 = arith.cmpi sge, %false, %true_0 : i1
    vector.print %0 : i1
    vector.print %1 : i1
    return
}

func.func @zero_ult_min_index() {
    // 0 `ult` -2^63 = true
    // CHECK-NEXT: 1
    %c0 = arith.constant 0 : index
    %c-9223372036854775808 = arith.constant -9223372036854775808 : index
    %0 = arith.cmpi ult, %c0, %c-9223372036854775808 : index
    vector.print %0 : i1
    return
}

func.func @entry() {
    func.call @signed_comparison_on_i1s() : () -> ()
    func.call @sge_0_1_is_true() : () -> ()
    func.call @zero_ult_min_index() : () -> ()
    return
}
