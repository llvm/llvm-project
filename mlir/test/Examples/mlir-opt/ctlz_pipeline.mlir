// RUN: mlir-opt --pass-pipeline='
// RUN:    builtin.module(
// RUN:        convert-math-to-funcs{convert-ctlz=1},
// RUN:        func.func(cse,canonicalize),
// RUN:        convert-scf-to-cf,
// RUN:        convert-to-llvm
// RUN:    )' %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK: llvm
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}

