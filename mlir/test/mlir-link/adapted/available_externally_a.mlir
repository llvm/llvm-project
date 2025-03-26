// RUN: mlir-link %s %p/available_externally_b.mlir -o - | FileCheck %s
// RUN: mlir-link %s -o - | FileCheck --check-prefix=AE-ONLY %s
module {
   llvm.mlir.global available_externally unnamed_addr constant @foo(0 : i32) {addr_space = 0 : i32} : i32
}

// CHECK: llvm.mlir.global external hidden unnamed_addr constant @foo(0 : i32)
// AE-ONLY-NOT: @foo
