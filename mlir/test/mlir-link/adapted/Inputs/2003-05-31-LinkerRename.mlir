module {
  llvm.mlir.global external @bar() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.addressof @foo : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func internal @foo() -> i32 attributes {dso_local} {
    %0 = llvm.mlir.constant(7 : i32) : i32
    llvm.return %0 : i32
  }
}
