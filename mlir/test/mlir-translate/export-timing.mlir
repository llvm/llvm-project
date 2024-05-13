// RUN: mlir-translate %s --mlir-to-llvmir -mlir-timing 2>&1 | FileCheck %s

// CHECK: Execution time report
// CHECK: Total Execution Time:
// CHECK: Name
// CHECK-NEXT: Translate MLIR to LLVMIR

llvm.func @foo() {
  llvm.return
}
