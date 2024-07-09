// Tests division operations and their variants (e.g. ceil/floordiv, rem etc)

// RUN: mlir-opt %s --arith-expand --test-lower-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @divsi_i8(%v1 : i8, %v2 : i8) {
    vector.print str "@divsi_i8\n"
    %0 = arith.divsi %v1, %v2 : i8
    vector.print %0 : i8
    return
}

func.func @divsi_i1(%v1 : i1, %v2 : i1) {
    vector.print str "@divsi_i1\n"
    %0 = arith.divsi %v1, %v2 : i1
    vector.print %0 : i1
    return

}

func.func @remsi_i8(%v1 : i8, %v2 : i8) {
    vector.print str "@remsi_i8\n"
    %0 = arith.remsi %v1, %v2 : i8
    vector.print %0 : i8
    return
}

func.func @ceildivsi_i8(%v1 : i8, %v2 : i8) {
    vector.print str "@ceildivsi_i8\n"
    %0 = arith.ceildivsi %v1, %v2 : i8
    vector.print %0 : i8
    return
}

func.func @divsi() {
    // ------------------------------------------------
    // Test i8
    // ------------------------------------------------
    %c68 = arith.constant 68 : i8
    %cn97 = arith.constant -97 : i8

    // divsi should round towards zero (rather than -infinity)
    // divsi -97 68 = -1
    // CHECK-LABEL: @divsi_i8
    // CHECK-NEXT:  -1
    func.call @divsi_i8(%cn97, %c68) : (i8, i8) -> ()

    // ------------------------------------------------
    // Test i1
    // ------------------------------------------------
    %false = arith.constant false
    %true = arith.constant true

    // CHECK-LABEL: @divsi_i1
    // CHECK-NEXT:  1
    func.call @divsi_i1(%true, %true) : (i1, i1) -> ()

    // ------------------------------------------------
    // TODO: i16, i32 etc
    // ------------------------------------------------   
    return 
}

func.func @remsi() {
    // ------------------------------------------------
    // Test i8
    // ------------------------------------------------
    %cn1 = arith.constant -1 : i8
    %i8_min_p1 = arith.constant -127 : i8
    
    // remsi minIntPlus1 -1 = remsi -2^(w-1) -1 = 0
    // CHECK-LABEL: @remsi_i8
    // CHECK-NEXT:  0
    func.call @remsi_i8(%i8_min_p1, %cn1) : (i8, i8) -> ()

    // ------------------------------------------------
    // TODO: i1, i16 etc
    // ------------------------------------------------ 
    return
}

func.func @ceildivsi() {
    // ------------------------------------------------
    // Test i8
    // ------------------------------------------------
    %c7 = arith.constant 7 : i8
    %i8_min = arith.constant -128 : i8

    // ceildivsi should keep signs
    // forall w, y. (w > 0, y > 0) => -2^w `ceildiv` y : i_w < 0
    // CHECK-LABEL: @ceildivsi_i8
    // CHECK-NEXT:  -18
    func.call @ceildivsi_i8(%i8_min, %c7) : (i8, i8) -> ()

    // ------------------------------------------------
    // TODO: i1, i16 etc
    // ------------------------------------------------ 

    return
}

func.func @entry() {
    func.call @divsi() : () -> ()
    func.call @remsi() : () -> ()
    func.call @ceildivsi() : () -> ()
    return
}
