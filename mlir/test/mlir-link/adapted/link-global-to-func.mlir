// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// edge case of linking function to global not yet supported
// XFAIL: *

// CHECK: __eprintf
module {
  llvm.mlir.global external @__eprintf() {addr_space = 0 : i32} : !llvm.ptr
  llvm.func @test() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @__eprintf : !llvm.ptr
    %1 = llvm.load %0 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
}

// -----

module {
  llvm.func @__eprintf(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) attributes {passthrough = ["noreturn"]}
  llvm.func @foo() {
    %0 = llvm.mlir.undef : !llvm.ptr
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.zero : !llvm.ptr
    llvm.call tail @__eprintf(%0, %0, %1, %2) {no_unwind} : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> ()
    llvm.unreachable
  }
}

