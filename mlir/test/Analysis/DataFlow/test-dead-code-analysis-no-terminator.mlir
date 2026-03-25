// RUN: mlir-opt --sccp %s | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/188408
// DeadCodeAnalysis crashed when visiting a block inside a region with the
// NoTerminator trait (e.g. acc.kernel_environment) because
// isRegionOrCallableReturn called Block::getTerminator() without first
// checking Block::mightHaveTerminator().

// CHECK-LABEL: func.func @f
func.func @f(%arg0: memref<8xi32>) {
  acc.kernel_environment {
    acc.compute_region {
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}
