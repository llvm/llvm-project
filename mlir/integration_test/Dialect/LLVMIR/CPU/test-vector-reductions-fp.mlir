// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// End-to-end test of all fp reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @print_newline()
  llvm.func @print_f32(!llvm.float)
  llvm.func @entry() {
    // Setup (1,2,3,4).
    %0 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
    %1 = llvm.mlir.constant(2.000000e+00 : f32) : !llvm.float
    %2 = llvm.mlir.constant(3.000000e+00 : f32) : !llvm.float
    %3 = llvm.mlir.constant(4.000000e+00 : f32) : !llvm.float
    %4 = llvm.mlir.undef : !llvm<"<4 x float>">
    %5 = llvm.mlir.constant(0 : index) : !llvm.i64
    %6 = llvm.insertelement %0, %4[%5 : !llvm.i64] : !llvm<"<4 x float>">
    %7 = llvm.shufflevector %6, %4 [0 : i32, 0 : i32, 0 : i32, 0 : i32]
        : !llvm<"<4 x float>">, !llvm<"<4 x float>">
    %8 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %9 = llvm.insertelement %1, %7[%8 : !llvm.i64] : !llvm<"<4 x float>">
    %10 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %11 = llvm.insertelement %2, %9[%10 : !llvm.i64] : !llvm<"<4 x float>">
    %12 = llvm.mlir.constant(3 : i64) : !llvm.i64
    %v = llvm.insertelement %3, %11[%12 : !llvm.i64] : !llvm<"<4 x float>">

    %max = "llvm.intr.experimental.vector.reduce.fmax"(%v)
        : (!llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%max) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 4

    %min = "llvm.intr.experimental.vector.reduce.fmin"(%v)
        : (!llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%min) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 1

    %add1 = "llvm.intr.experimental.vector.reduce.v2.fadd"(%0, %v)
        : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%add1) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 11

    %add1r = "llvm.intr.experimental.vector.reduce.v2.fadd"(%0, %v)
        {reassoc = true} : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%add1r) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 11

    %add2 = "llvm.intr.experimental.vector.reduce.v2.fadd"(%1, %v)
        : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%add2) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 12

    %add2r = "llvm.intr.experimental.vector.reduce.v2.fadd"(%1, %v)
        {reassoc = true} : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%add2r) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 12

    %mul1 = "llvm.intr.experimental.vector.reduce.v2.fmul"(%0, %v)
        : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%mul1) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 24

    %mul1r = "llvm.intr.experimental.vector.reduce.v2.fmul"(%0, %v)
        {reassoc = true} : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%mul1r) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 24

    %mul2 = "llvm.intr.experimental.vector.reduce.v2.fmul"(%1, %v)
        : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%mul2) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 48

    %mul2r = "llvm.intr.experimental.vector.reduce.v2.fmul"(%1, %v)
        {reassoc = true} : (!llvm.float, !llvm<"<4 x float>">) -> !llvm.float
    llvm.call @print_f32(%mul2r) : (!llvm.float) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 48

    llvm.return
  }
}
