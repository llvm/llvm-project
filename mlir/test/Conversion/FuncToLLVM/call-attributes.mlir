// RUN: mlir-opt -convert-func-to-llvm %s | FileCheck %s

func.func private @callee(f32) -> f32

// CHECK-LABEL: llvm.func @direct_call
func.func @direct_call(%arg: f32) -> f32 {
  // CHECK: llvm.call fastcc @callee(%arg0) {convergent, fastmathFlags = #llvm.fastmath<fast>, test.marker = "keep"}
  %0 = func.call @callee(%arg) {CConv = #llvm.cconv<fastcc>, fastmathFlags = #llvm.fastmath<fast>, convergent, test.marker = "keep"} : (f32) -> f32
  return %0 : f32
}

// CHECK-LABEL: llvm.func @indirect_call
func.func @indirect_call(%fn: (f32) -> f32, %arg: f32) -> f32 {
  // CHECK: llvm.call fastcc %arg0(%arg1) {convergent, fastmathFlags = #llvm.fastmath<nnan>, test.marker = "keep"}
  %0 = func.call_indirect %fn(%arg) {CConv = #llvm.cconv<fastcc>, fastmathFlags = #llvm.fastmath<nnan>, convergent, test.marker = "keep"} : (f32) -> f32
  return %0 : f32
}
