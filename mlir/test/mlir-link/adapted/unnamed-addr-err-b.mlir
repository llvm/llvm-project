// RUN: true

module {
  llvm.mlir.global appending @foo(dense<42> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>
}
