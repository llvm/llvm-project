// RUN: echo "module {}" > %t.LinkTest.mlir
// RUN: mlir-link %t.LinkTest.mlir %s

module {
  llvm.mlir.global external @work(4 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global external @test() {addr_space = 0 : i32} : !llvm.ptr {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.addressof @work : !llvm.ptr
    %2 = llvm.getelementptr %1[%0] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return %2 : !llvm.ptr
  }
}
