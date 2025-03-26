// RUN: echo "module { llvm.mlir.global appending @X(dense<8> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32> }" > %t.tmp.mlir
// RUN: mlir-link %s %t.tmp.mlir | FileCheck %s

// appending linkage not yet implemented
// XFAIL: *

// CHECK: dense<[7, 8]>

module {
  llvm.mlir.global appending @X(dense<7> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>
}
