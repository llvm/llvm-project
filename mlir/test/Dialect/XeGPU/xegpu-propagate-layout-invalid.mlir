// RUN: mlir-opt --xegpu-propagate-layout -split-input-file %s | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/177846:
// --xegpu-propagate-layout must not crash when the module contains an
// llvm.func declaration. updateFunctionOpInterface called setType(FunctionType)
// on an llvm.func (whose type is LLVMFunctionType), corrupting its
// function_type attribute; the subsequent getFunctionType() then
// triggered cast<LLVMFunctionType> on a FunctionType and aborted.

// CHECK-LABEL: llvm.func @some_function()
module {
  llvm.func @some_function()
}

// -----

// CHECK-LABEL: transform.named_sequence @callee(!transform.any_op {transform.readonly})
module attributes {transform.with_named_sequence} {
  transform.named_sequence @callee(!transform.any_op {transform.readonly})
  transform.named_sequence @caller(%arg0: !transform.any_op {transform.readonly}) {
    transform.include @callee failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func private @declaration(i32) -> i32
func.func private @declaration(i32) -> i32
