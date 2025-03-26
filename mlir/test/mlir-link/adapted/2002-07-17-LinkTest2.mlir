// RUN: echo "module {}" > %t1.mlir
// RUN: mlir-link %t1.mlir %s

module {
  llvm.mlir.global external @work() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @zip : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @zip(i32, i32) -> i32
}
