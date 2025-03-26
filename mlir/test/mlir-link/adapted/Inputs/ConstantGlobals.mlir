module {
  llvm.mlir.global external constant @X(dense<8> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>
  llvm.mlir.global external constant @Y() {addr_space = 0 : i32} : !llvm.array<1 x i32>
}
