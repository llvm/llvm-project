// RUN: mlir-opt --pass-pipeline=' builtin.module( convert-math-to-funcs{convert-ctlz=1}, func.func(cse,canonicalize), convert-scf-to-cf, convert-to-llvm)' %s | FileCheck %s

// CHECK-LABEL: @main
// CHECK: llvm
func.func @main(%arg0: i32) -> i32 {
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}

