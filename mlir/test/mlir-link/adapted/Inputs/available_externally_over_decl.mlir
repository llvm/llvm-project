module {
  llvm.mlir.global external @h() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @f : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.mlir.global external @h2() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @g : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func available_externally @f() {
    llvm.return
  }
  llvm.func available_externally @g() {
    llvm.return
  }
}
