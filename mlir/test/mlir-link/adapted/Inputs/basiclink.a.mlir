module {
  llvm.mlir.global external @baz() {addr_space = 0 : i32} : i32
  llvm.func @foo(%arg0: i32) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @baz : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
