// RUN: mlir-opt --mlir-very-unsafe-disable-verifier-on-parsing %s | FileCheck %s

// CHECK: // 'builtin.module' failed to verify and will be printed in generic form
func.func @foo() -> tensor<10xi32> { return }
