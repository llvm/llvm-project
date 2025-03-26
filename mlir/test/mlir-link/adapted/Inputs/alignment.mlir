module {
  llvm.mlir.global external @A(7 : i32) {addr_space = 0 : i32, alignment = 8 : i64} : i32
  llvm.mlir.global external @B(7 : i32) {addr_space = 0 : i32, alignment = 4 : i64} : i32
  llvm.func @C() attributes {alignment = 8 : i64} {
    llvm.return
  }
  llvm.func @D() attributes {alignment = 4 : i64} {
    llvm.return
  }
  llvm.mlir.global common @E(0 : i32) {addr_space = 0 : i32, alignment = 8 : i64} : i32
}
