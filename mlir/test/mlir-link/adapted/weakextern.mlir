// RUN: mlir-link %s %s %p/testlink.mlir -o - | FileCheck %s

// CHECK-DAG: llvm.mlir.global extern_weak @kallsyms_names
// CHECK-DAG: llvm.mlir.global external @Inte
// CHECK-DAG: llvm.mlir.global external @MyVar

module {
  llvm.mlir.global extern_weak @kallsyms_names() {addr_space = 0 : i32} : !llvm.array<0 x i8>
  llvm.mlir.global extern_weak @MyVar() {addr_space = 0 : i32} : i32
  llvm.mlir.global extern_weak @Inte() {addr_space = 0 : i32} : i32
  llvm.func weak @use_kallsyms_names() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @kallsyms_names : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
