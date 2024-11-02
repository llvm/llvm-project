// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// End-to-end test of all int reduction intrinsics (not exhaustive unit tests).
module {
  llvm.func @printNewline()
  llvm.func @printI64(i64)
  llvm.func @entry() {
    // Setup (1,2,3,4).
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(2 : i64) : i64
    %2 = llvm.mlir.constant(3 : i64) : i64
    %3 = llvm.mlir.constant(4 : i64) : i64
    %4 = llvm.mlir.undef : vector<4xi64>
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.insertelement %0, %4[%5 : i64] : vector<4xi64>
    %7 = llvm.shufflevector %6, %4 [0, 0, 0, 0] : vector<4xi64>
    %8 = llvm.mlir.constant(1 : i64) : i64
    %9 = llvm.insertelement %1, %7[%8 : i64] : vector<4xi64>
    %10 = llvm.mlir.constant(2 : i64) : i64
    %11 = llvm.insertelement %2, %9[%10 : i64] : vector<4xi64>
    %12 = llvm.mlir.constant(3 : i64) : i64
    %v = llvm.insertelement %3, %11[%12 : i64] : vector<4xi64>

    %add = "llvm.intr.vector.reduce.add"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%add) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 10

    %and = "llvm.intr.vector.reduce.and"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%and) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 0

    %mul = "llvm.intr.vector.reduce.mul"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%mul) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 24

    %or = "llvm.intr.vector.reduce.or"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%or) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 7

    %smax = "llvm.intr.vector.reduce.smax"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%smax) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %smin = "llvm.intr.vector.reduce.smin"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%smin) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %umax = "llvm.intr.vector.reduce.umax"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%umax) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    %umin = "llvm.intr.vector.reduce.umin"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%umin) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1

    %xor = "llvm.intr.vector.reduce.xor"(%v)
        : (vector<4xi64>) -> i64
    llvm.call @printI64(%xor) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 4

    llvm.return
  }
}
