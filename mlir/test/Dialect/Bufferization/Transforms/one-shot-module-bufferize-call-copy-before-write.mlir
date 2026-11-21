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

// -----

// Regression test for https://github.com/llvm/llvm-project/issues/217227:
// this function does not have a `func.return` terminator (it uses
// `spirv.Return` instead, which is a valid terminator for `func.func` since
// the verifier only requires the region to end in some terminator, not
// specifically a `func.return`). One-Shot Module Bufferize must not crash on
// such IR; it has no tensors to bufferize, so it should simply leave the
// function unchanged.

// CHECK-LABEL: func.func private @non_func_return(
// CHECK-SAME:      %{{.*}}: memref<4xi32>)
// CHECK-NEXT:    spirv.Return
func.func private @non_func_return(%arg0: memref<4xi32>) {
  spirv.Return
}
