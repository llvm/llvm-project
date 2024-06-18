// RUN: mlir-opt --convert-math-to-llvm %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK-NEXT: llvm.intr.ctlz
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
