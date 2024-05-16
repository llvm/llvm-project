// tests arith operations on i1 type.

// RUN: mlir-opt %s --convert-scf-to-cf --convert-cf-to-llvm --convert-vector-to-llvm \
// RUN:             --convert-func-to-llvm --convert-arith-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @zeroPlusOneOnI1() {
    // addi on i1
    // addi(0, 1) : i1 = 1 : i1; addi(0, -1) : i1 = 1
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

func.func @i1Printing() {
    // printing i1 values
    // print(0 : i1) = '0'; print(1 : i1) = '1'; print(-1 : i1) = '1'
    %false = arith.constant false
    %true = arith.constant 1 : i1
    %true_0 = arith.constant -1 : i1
    vector.print %false : i1
    vector.print %true : i1
    vector.print %true_0 : i1
    return
}

func.func @signedComparisonOnI1s() {
    // signed comparisons on i1s
    // slt 0 1 = false, sle 0 1 = false, sgt 0 1 = true, sge 0 1 = true
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

func.func @sge0And1IsTrue() {
    // sge 0 -1, sge 0 1, should be true
    // sge 0 -1 == sge 0 1 == true (1)
    %false = arith.constant 0 : i1
    %true = arith.constant 1 : i1
    %true_0 = arith.constant -1 : i1
    %0 = arith.cmpi sge, %false, %true : i1
    %1 = arith.cmpi sge, %false, %true_0 : i1
    vector.print %0 : i1
    vector.print %1 : i1
    return
}

func.func @divsiI1SignedRepr() {
    // divsi should output unsigned representation, e.g. in the case of i1
    // repr (divsi (x : i1) (x : i1)) = 1 (not represented as -1)
    %false = arith.constant false
    %true = arith.constant true
    %0 = arith.divsi %true, %true : i1
    vector.print str "%2="
    vector.print %0 : i1
    %1 = arith.cmpi sge, %false, %0 : i1
    vector.print str "%3="
    vector.print %1 : i1
    return
}

func.func @adduiExtendedI1() {
    // addui_extended on i1
    // addui_extended 1 1 : i1 = 0, 1
    %true = arith.constant 1 : i1
    %sum, %overflow = arith.addui_extended %true, %true : i1, i1
    vector.print %sum : i1
    vector.print %overflow : i1
    return
}

func.func @adduiExtendedOverflowBitIsN1() {
    // addui_extended overflow bit is treated as -1
    // addui_extended -1633386 -1643386 = ... 1 (overflow because negative numbers are large positive numbers)
    %c-16433886_i64 = arith.constant -16433886 : i64
    %sum, %overflow = arith.addui_extended %c-16433886_i64, %c-16433886_i64 : i64, i1
    %false = arith.constant false
    %0 = arith.cmpi sge, %overflow, %false : i1
    vector.print %0 : i1 // but prints as "1"
    return
}

func.func @mulsiExtendedOnI1() {
    // mulsi_extended on i1, tests for overflow bit
    // mulsi_extended 1, 1 : i1 = (1, 0)
    %true = arith.constant true
    %low, %high = arith.mulsi_extended %true, %true : i1
    vector.print %low : i1
    vector.print %high : i1
    return
}

func.func @entry() {
    // CHECK:      1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    func.call @zeroPlusOneOnI1() : () -> ()

    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    func.call @i1Printing() : () -> ()

    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    func.call @signedComparisonOnI1s() : () -> ()

    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    func.call @sge0And1IsTrue() : () -> ()

    // CHECK-NEXT: %2=1
    // CHECK-NEXT: %3=1
    func.call @divsiI1SignedRepr() : () -> ()

    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    func.call @adduiExtendedI1() : () -> ()

    // CHECK-NEXT: 0
    func.call @adduiExtendedOverflowBitIsN1() : () -> ()

    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    func.call @mulsiExtendedOnI1() : () -> ()

    return
}
