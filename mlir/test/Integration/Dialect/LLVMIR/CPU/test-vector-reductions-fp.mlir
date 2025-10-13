// RUN: mlir-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// End-to-end test of all fp reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @printNewline()
  llvm.func @printF32(f32)
  llvm.func @entry() {
    // Setup (1,2,3,4).
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %1 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %2 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %3 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %4 = llvm.mlir.undef : vector<4xf32>
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.insertelement %0, %4[%5 : i64] : vector<4xf32>
    %7 = llvm.shufflevector %6, %4 [0, 0, 0, 0] : vector<4xf32>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.insertelement %1, %7[%8 : i64] : vector<4xf32>
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.insertelement %2, %9[%10 : i64] : vector<4xf32>
    %12 = llvm.mlir.constant(3 : i64) : i64
    %v = llvm.insertelement %3, %11[%12 : i64] : vector<4xf32>

    %max = llvm.intr.vector.reduce.fmax(%v)
        : (vector<4xf32>) -> f32
    llvm.call @printF32(%max) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %min = llvm.intr.vector.reduce.fmin(%v)
        : (vector<4xf32>) -> f32
    llvm.call @printF32(%min) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %maximum = llvm.intr.vector.reduce.fmaximum(%v)
        : (vector<4xf32>) -> f32
    llvm.call @printF32(%maximum) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %minimum = llvm.intr.vector.reduce.fminimum(%v)
        : (vector<4xf32>) -> f32
    llvm.call @printF32(%minimum) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %add1 = "llvm.intr.vector.reduce.fadd"(%0, %v)
        : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%add1) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 11

    %add1r = "llvm.intr.vector.reduce.fadd"(%0, %v)
        {reassoc = true} : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%add1r) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 11

    %add2 = "llvm.intr.vector.reduce.fadd"(%1, %v)
        : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%add2) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 12

    %add2r = "llvm.intr.vector.reduce.fadd"(%1, %v)
        {reassoc = true} : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%add2r) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 12

    %mul1 = "llvm.intr.vector.reduce.fmul"(%0, %v)
        : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%mul1) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 24

    %mul1r = "llvm.intr.vector.reduce.fmul"(%0, %v)
        {reassoc = true} : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%mul1r) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 24

    %mul2 = "llvm.intr.vector.reduce.fmul"(%1, %v)
        : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%mul2) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 48

    %mul2r = "llvm.intr.vector.reduce.fmul"(%1, %v)
        {reassoc = true} : (f32, vector<4xf32>) -> f32
    llvm.call @printF32(%mul2r) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 48

    llvm.return
  }
}
