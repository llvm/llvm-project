// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @extsi_i1_i16(%v1 : i1) {
    vector.print str "@extsi_i1_i16\n"
    %0 = arith.extsi %v1 : i1 to i16
    vector.print %0 : i16
    return
}

func.func @extui_i1_i64(%v1 : i1) {
    vector.print str "@extui_i1_i64\n"
    %0 = arith.extui %v1 : i1 to i64
    vector.print %0 : i64
    return
}

func.func @trunci_i16_i8(%v1 : i16) {
    vector.print str "@trunci_i16_i8\n"
    %0 = arith.trunci %v1 : i16 to i8
    vector.print %0 : i8
    return
}

func.func @extsi() {
    // ------------------------------------------------
    // Test extending from i1
    // ------------------------------------------------
    %true = arith.constant -1 : i1

    // extsi on 1 : i1
    // extsi(1: i1) = -1 : i16
    // CHECK-LABEL: @extsi_i1_i16
    // CHECK-NEXT:  -1
    func.call @extsi_i1_i16(%true) : (i1) -> ()

    // ------------------------------------------------
    // TODO: Test extension from i8, i16 etc..
    // ------------------------------------------------

    return
}

func.func @extui() {
    // ------------------------------------------------
    // Test extending from i1
    // ------------------------------------------------
    %true = arith.constant true

    // extui should extend i1 with 0 bits not 1s
    // extui(1 : i1) = 1 : i64
    // CHECK-LABEL: @extui_i1_i64
    // CHECK-NEXT:  1
    func.call @extui_i1_i64(%true) : (i1) -> ()

    // ------------------------------------------------
    // TODO: Test extension from i8, i16 etc..
    // ------------------------------------------------

    return
}

func.func @trunci() {
    // ------------------------------------------------
    // Test truncating from i16
    // ------------------------------------------------

    // trunci on 20194 : i16
    // trunci(20194 : i16) = -30 : i8
    // CHECK-LABEL: @trunci_i16_i8
    // CHECK-NEXT:  -30
    %c20194 = arith.constant 20194 : i16
    func.call @trunci_i16_i8(%c20194) : (i16) -> ()

    // ------------------------------------------------
    // TODO: Test truncation of i1, i8 etc..
    // ------------------------------------------------

    return
}

func.func @entry() {
    func.call @extsi() : () -> ()
    func.call @extui() : () -> ()
    func.call @trunci() : () -> ()
    return
}
