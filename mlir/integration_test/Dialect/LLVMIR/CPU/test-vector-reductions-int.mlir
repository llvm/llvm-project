// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// End-to-end test of all int reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @print_newline()
  llvm.func @print_i32(!llvm.i32)
  llvm.func @entry() {
    // Setup (1,2,3,4).
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %4 = llvm.mlir.undef : !llvm<"<4 x i32>">
    %5 = llvm.mlir.constant(0 : index) : !llvm.i64
    %6 = llvm.insertelement %0, %4[%5 : !llvm.i64] : !llvm<"<4 x i32>">
    %7 = llvm.shufflevector %6, %4 [0 : i32, 0 : i32, 0 : i32, 0 : i32]
        : !llvm<"<4 x i32>">, !llvm<"<4 x i32>">
    %8 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %9 = llvm.insertelement %1, %7[%8 : !llvm.i64] : !llvm<"<4 x i32>">
    %10 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %11 = llvm.insertelement %2, %9[%10 : !llvm.i64] : !llvm<"<4 x i32>">
    %12 = llvm.mlir.constant(3 : i64) : !llvm.i64
    %v = llvm.insertelement %3, %11[%12 : !llvm.i64] : !llvm<"<4 x i32>">

    %add = "llvm.intr.experimental.vector.reduce.add"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%add) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 10

    %and = "llvm.intr.experimental.vector.reduce.and"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%and) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 0

    %mul = "llvm.intr.experimental.vector.reduce.mul"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%mul) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 24

    %or = "llvm.intr.experimental.vector.reduce.or"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%or) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 7

    %smax = "llvm.intr.experimental.vector.reduce.smax"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%smax) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 4

    %smin = "llvm.intr.experimental.vector.reduce.smin"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%smin) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 1

    %umax = "llvm.intr.experimental.vector.reduce.umax"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%umax) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 4

    %umin = "llvm.intr.experimental.vector.reduce.umin"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%umin) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 1

    %xor = "llvm.intr.experimental.vector.reduce.xor"(%v)
        : (!llvm<"<4 x i32>">) -> !llvm.i32
    llvm.call @print_i32(%xor) : (!llvm.i32) -> ()
    llvm.call @print_newline() : () -> ()
    // CHECK: 4

    llvm.return
  }
}
