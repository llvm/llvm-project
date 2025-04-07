// RUN: not mlir-link %s %S/unnamed-addr-err-b.mlir 2>&1 | FileCheck %s

// CHECK: Appending variables with different unnamed_addr need to be linked

module {
  llvm.mlir.global appending unnamed_addr @foo(dense<42> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>
}
