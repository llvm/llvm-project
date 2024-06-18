// RUN: mlir-opt --convert-math-to-funcs=convert-ctlz %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK-NEXT: call @__mlir_math_ctlz_i32

// CHECK-LABEL: func.func private @__mlir_math_ctlz_i32
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
