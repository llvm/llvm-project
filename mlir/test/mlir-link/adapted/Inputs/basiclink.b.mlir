module {
  llvm.mlir.global external @baz(0 : i32) {addr_space = 0 : i32} : i32
  llvm.func @foo(...) -> !llvm.ptr
  llvm.func @bar() -> !llvm.ptr {
    %0 = llvm.mlir.constant(123 : i32) : i32
    %1 = llvm.call @foo(%0) vararg(!llvm.func<ptr (...)>) : (i32) -> !llvm.ptr
    llvm.return %1 : !llvm.ptr
  }
}
