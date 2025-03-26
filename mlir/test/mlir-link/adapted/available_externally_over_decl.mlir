// RUN: mlir-link %s %p/Inputs/available_externally_over_decl.mlir | FileCheck %s
module {
  llvm.func @f()
  llvm.func available_externally @g() {
    llvm.return
  }
  llvm.func @main() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @f : !llvm.ptr
    llvm.call @g() : () -> ()
    llvm.return %0 : !llvm.ptr
  }
}

// CHECK-DAG: llvm.func available_externally @g() {
// CHECK-DAG: llvm.func available_externally @f() {
