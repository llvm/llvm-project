// RUN: mlir-opt --convert-cf-to-llvm %s | FileCheck %s

// CHECK-LABEL: @assume
// CHECK-SAME: %[[ARG:.+]]:
func.func @assume(%arg: i1) {
  // CHECK: llvm.intr.assume %[[ARG]]
  cf.assume %arg
  return
}
