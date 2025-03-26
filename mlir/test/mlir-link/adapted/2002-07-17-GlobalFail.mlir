// RUN: echo "module {}" > %t.tmp.mlir
// RUN: mlir-link %t.tmp.mlir %s
module {
  llvm.mlir.global external constant @X(5 : i32) {addr_space = 0 : i32} : i32
  llvm.mlir.global internal @Y() {addr_space = 0 : i32, dso_local} : !llvm.array<2 x ptr> {
    %0 = llvm.mlir.addressof @X : !llvm.ptr
    %1 = llvm.mlir.undef : !llvm.array<2 x ptr>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.array<2 x ptr>
    %3 = llvm.insertvalue %0, %2[1] : !llvm.array<2 x ptr>
    llvm.return %3 : !llvm.array<2 x ptr>
  }
}
