module {
  llvm.mlir.global linkonce @X(5 : i32) {addr_space = 0 : i32} : i32
  llvm.func linkonce @foo() -> i32 {
    %0 = llvm.mlir.constant(7 : i32) : i32
    llvm.return %0 : i32
  }
}
