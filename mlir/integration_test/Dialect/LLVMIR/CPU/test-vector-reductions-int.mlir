// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// End-to-end test of all int reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @printNewline()
  llvm.func @printI64(!llvm.i64)
  llvm.func @entry() {
    // Setup (1,2,3,4).
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %2 = llvm.mlir.constant(3 : i64) : !llvm.i64
    %3 = llvm.mlir.constant(4 : i64) : !llvm.i64
    %4 = llvm.mlir.undef : !llvm.vec<4 x i64>
    %5 = llvm.mlir.constant(0 : index) : !llvm.i64
    %6 = llvm.insertelement %0, %4[%5 : !llvm.i64] : !llvm.vec<4 x i64>
    %7 = llvm.shufflevector %6, %4 [0 : i64, 0 : i64, 0 : i64, 0 : i64]
        : !llvm.vec<4 x i64>, !llvm.vec<4 x i64>
    %8 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %9 = llvm.insertelement %1, %7[%8 : !llvm.i64] : !llvm.vec<4 x i64>
    %10 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %11 = llvm.insertelement %2, %9[%10 : !llvm.i64] : !llvm.vec<4 x i64>
    %12 = llvm.mlir.constant(3 : i64) : !llvm.i64
    %v = llvm.insertelement %3, %11[%12 : !llvm.i64] : !llvm.vec<4 x i64>

    %add = "llvm.intr.vector.reduce.add"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%add) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 10

    %and = "llvm.intr.vector.reduce.and"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%and) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 0

    %mul = "llvm.intr.vector.reduce.mul"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%mul) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 24

    %or = "llvm.intr.vector.reduce.or"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%or) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 7

    %smax = "llvm.intr.vector.reduce.smax"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%smax) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %smin = "llvm.intr.vector.reduce.smin"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%smin) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %umax = "llvm.intr.vector.reduce.umax"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%umax) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %umin = "llvm.intr.vector.reduce.umin"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%umin) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %xor = "llvm.intr.vector.reduce.xor"(%v)
        : (!llvm.vec<4 x i64>) -> !llvm.i64
    llvm.call @printI64(%xor) : (!llvm.i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    llvm.return
  }
}
