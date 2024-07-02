// Tests division operations and their variants (e.g. ceil/floordiv, rem etc)
// These tests are intended to be target agnostic: they should yield the same results 
// regardless of the target platform.

// RUN: mlir-opt %s --test-lower-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @divsi_round_towards_zero() {
    // divsi should round towards zero (rather than -infinity)
    // divsi -97 68 = -1
    // CHECK: -1
    %c68_i8 = arith.constant 68 : i8
    %c-97_i8 = arith.constant -97 : i8
    %0 = arith.divsi %c-97_i8, %c68_i8 : i8
    vector.print %0 : i8
    return
}

func.func @remsi_print_zero() {
    // remsi minInt -1 = 0
    // remsi -2^(w-1) -1 = 0
    // CHECK-NEXT:  -1
    // CHECK-NEXT:  -128
    // CHECK-NEXT:  0
    %c-1_i8 = arith.constant -1 : i8
    %c-128_i8 = arith.constant -128 : i8
    %0 = arith.remsi %c-128_i8, %c-1_i8 : i8
    vector.print %c-1_i8 : i8
    vector.print %c-128_i8 : i8
    vector.print %0 : i8
    return
}

func.func @ceildivsi_keep_signs() {
    // ceildivsi should keep signs
    // forall w, y. (w > 0, y > 0) => -2^w `ceildiv` y : i_w < 0
    // CHECK-NEXT:  -18
    %c7_i8 = arith.constant 7 : i8
    %c-128_i8 = arith.constant -128 : i8
    %0 = arith.ceildivsi %c-128_i8, %c7_i8 : i8
    vector.print %0 : i8
    return
}

func.func @divsi_i1_signed_repr() {
    // divsi should output unsigned representation, e.g. in the case of i1
    // repr (divsi (x : i1) (x : i1)) = 1 (not represented as -1)
    // CHECK-NEXT: %2=1
    // CHECK-NEXT: %3=1
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

func.func @entry() {
    func.call @divsi_round_towards_zero() : () -> ()
    func.call @remsi_print_zero() : () -> ()
    func.call @ceildivsi_keep_signs() : () -> ()
    func.call @divsi_i1_signed_repr() : () -> ()
    return
}
