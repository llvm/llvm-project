// tests simple arithmetic operations (i.e. add/sub/mul/div) and their variants (e.g. signed/unsigned, floor/ceildiv)

// RUN: mlir-opt %s --test-lower-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @divsiRoundTowardsZero() {
    // divsi should round towards zero (rather than -infinity)
    // divsi -97 68 = -1
    %c68_i8 = arith.constant 68 : i8
    %c-97_i8 = arith.constant -97 : i8
    %0 = arith.divsi %c-97_i8, %c68_i8 : i8
    vector.print %0 : i8
    return
}

func.func @mulsimuluiExtendedOverflows() {
    // mulsi and mului extended versions, with overflow
    // mulsi_extended -100, -100 : i8 = (16, 39); mului_extended -100, -100 : i8 = (16, 95)
    %c-100_i8 = arith.constant -100 : i8
    %low, %high = arith.mulsi_extended %c-100_i8, %c-100_i8 : i8
    vector.print %low : i8
    vector.print %high : i8
    %low_0, %high_1 = arith.mului_extended %c-100_i8, %c-100_i8 : i8
    vector.print %low_0 : i8
    vector.print %high_1 : i8
    return
}

func.func @remsiPrintZero() {
    // remsi minInt -1 = 0
    // remsi -2^(w-1) -1 = 0
    %c-1_i8 = arith.constant -1 : i8
    %c-128_i8 = arith.constant -128 : i8
    %0 = arith.remsi %c-128_i8, %c-1_i8 : i8
    vector.print %c-1_i8 : i8
    vector.print %c-128_i8 : i8
    vector.print %0 : i8
    return
}

func.func @ceildivsiKeepSigns() {
    // ceildivsi should keep signs
    // forall w, y. (w > 0, y > 0) => -2^w `ceildiv` y : i_w < 0
    %c7_i8 = arith.constant 7 : i8
    %c-128_i8 = arith.constant -128 : i8
    %0 = arith.ceildivsi %c-128_i8, %c7_i8 : i8
    vector.print %0 : i8
    return
}

func.func @entry() {

    // CHECK:       -1
    func.call @divsiRoundTowardsZero() : () -> ()

    // CHECK-NEXT:  16
    // CHECK-NEXT:  39
    // CHECK-NEXT:  16
    // CHECK-NEXT:  95
    func.call @mulsimuluiExtendedOverflows() : () -> ()

    // CHECK-NEXT:  -1
    // CHECK-NEXT:  -128
    // CHECK-NEXT:  0
    func.call @remsiPrintZero() : () -> ()

    // CHECK-NEXT:  -18
    func.call @ceildivsiKeepSigns() : () -> ()
    
    return
}
