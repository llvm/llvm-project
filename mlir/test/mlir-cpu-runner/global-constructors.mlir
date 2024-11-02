// UNSUPPORTED: target=aarch64{{.*}}, target=arm64{{.*}}
// RUN: mlir-cpu-runner %s -e entry -entry-point-result=void  \
// RUN: -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

// Test that the `ctor` executes before `entry` and that `dtor` executes last.
module {
  llvm.func @printNewline()
  llvm.func @printI64(i64)
  llvm.mlir.global_ctors {ctors = [@ctor], priorities = [0 : i32]}
  llvm.mlir.global_dtors {dtors = [@dtor], priorities = [0 : i32]}
  llvm.func @ctor() {
    %0 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @printI64(%0) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 1
    llvm.return
  }
  llvm.func @entry() {
    %0 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @printI64(%0) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 2
    llvm.return
  }
  llvm.func @dtor() {
    %0 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @printI64(%0) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    // CHECK: 3
    llvm.return
  }
}
