module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo largest
    llvm.comdat_selector @qux largest
    llvm.comdat_selector @any any
  }
  llvm.mlir.global external @foo(43 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64
  llvm.mlir.global external @qux(13 : i32) comdat(@__llvm_global_comdat::@qux) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @in_unselected_group(13 : i32) comdat(@__llvm_global_comdat::@qux) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @any(7 : i64) comdat(@__llvm_global_comdat::@any) {addr_space = 0 : i32} : i64
  llvm.func @bar() -> i32 comdat(@__llvm_global_comdat::@foo) {
    %0 = llvm.mlir.constant(43 : i32) : i32
    llvm.return %0 : i32
  }
  llvm.func @baz() -> i32 comdat(@__llvm_global_comdat::@qux) {
    %0 = llvm.mlir.constant(13 : i32) : i32
    llvm.return %0 : i32
  }
}
