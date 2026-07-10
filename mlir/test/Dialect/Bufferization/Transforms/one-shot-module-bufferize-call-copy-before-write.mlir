// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 copy-before-write=1" | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/163052
// copy-before-write=1 + bufferize-function-boundaries=1 with a call to a
// private (declaration-only) function used to crash with a stack overflow due
// to an invalid cast of AnalysisState to OneShotAnalysisState inside
// getCalledFunction().

// CHECK-LABEL: func.func private @callee(memref<64xf32
// CHECK-LABEL: func.func @caller
// CHECK:         call @callee
func.func private @callee(tensor<64xf32>)
func.func @caller(%A : tensor<64xf32>) {
  call @callee(%A) : (tensor<64xf32>) -> ()
  return
}
