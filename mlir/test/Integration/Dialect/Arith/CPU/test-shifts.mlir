// tests arith shifting operations.

// RUN: mlir-opt %s --test-lower-to-llvm | \
// RUN:   mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:                   --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

func.func @shrsiPreserveSigns() {
    // shrsi preserves signs
    // shrsi -10 7 : i8 = -1
    %c7_i8 = arith.constant 7 : i8
    %c-10_i8 = arith.constant -10 : i8
    %0 = arith.shrsi %c-10_i8, %c7_i8 : i8
    vector.print %c7_i8 : i8
    vector.print %c-10_i8 : i8
    vector.print %0 : i8
    return
}

func.func @shiftOnZero() {
    // shifts on zero is identity
    // shrsi 7 0 : i8 = 7; shrui -10 0 : i8 = -10; shli 7 0 : i8 = 7
    %c7_i8 = arith.constant 7 : i8
    %c-10_i8 = arith.constant -10 : i8
    %c0_i8 = arith.constant 0 : i8
    %0 = arith.shrsi %c7_i8, %c0_i8 : i8
    %1 = arith.shrui %c-10_i8, %c0_i8 : i8
    %2 = arith.shli %c7_i8, %c0_i8 : i8
    vector.print %0 : i8
    vector.print %1 : i8
    vector.print %2 : i8
    return
}

func.func @shiftOnZeroI1NonPoison() {
    // shift by zero : i1 should be non poison
    // sh{rsi, rui, li} 0 0 : i1 = 0
    %false = arith.constant 0 : i1
    %0 = arith.shrsi %false, %false : i1
    %1 = arith.shrui %false, %false : i1
    %2 = arith.shli %false, %false : i1
    vector.print %0 : i1
    vector.print %1 : i1
    vector.print %2 : i1
    return
}

func.func @cmpiUnsigned() {
    // cmpi on i8, unsigned flags tests
    // 1 `ult` 1 = false (0); 1 `ule` 1 = true (1); 1 `ugt` 1 = false (0); 1 `uge` 1 = true (1)
    %c1_i8 = arith.constant 1 : i8
    %0 = arith.cmpi ult, %c1_i8, %c1_i8 : i8
    %1 = arith.cmpi ule, %c1_i8, %c1_i8 : i8
    %2 = arith.cmpi ugt, %c1_i8, %c1_i8 : i8
    %3 = arith.cmpi uge, %c1_i8, %c1_i8 : i8
    vector.print %0 : i1
    vector.print %1 : i1
    vector.print %2 : i1
    vector.print %3 : i1
    return
}

func.func @shiftLeftValueGoesIntoTheVoid() {
    // shli on i8, value goes off into the void (overflow/modulus needed)
    // shli (-100), 7
    %c-100_i8 = arith.constant -100 : i8
    %c7_i8 = arith.constant 7 : i8
    %0 = arith.shli %c-100_i8, %c7_i8 : i8
    vector.print %c-100_i8 : i8
    vector.print %c7_i8 : i8
    vector.print %0 : i8
    return
}

func.func @entry() {
    // CHECK:       7
    // CHECK-NEXT:  -10
    // CHECK-NEXT:  -1
    func.call @shrsiPreserveSigns() : () -> ()

    // CHECK-NEXT:  7
    // CHECK-NEXT:  -10
    // CHECK-NEXT:  7
    func.call @shiftOnZero() : () -> ()

    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    // CHECK-NEXT:  0
    func.call @shiftOnZeroI1NonPoison() : () -> ()

    // CHECK-NEXT:  0
    // CHECK-NEXT:  1
    // CHECK-NEXT:  0
    // CHECK-NEXT:  1
    func.call @cmpiUnsigned() : () -> ()

    // CHECK-NEXT:  -100
    // CHECK-NEXT:  7
    // CHECK-NEXT:  0
    func.call @shiftLeftValueGoesIntoTheVoid() : () -> ()
    
    return
}
