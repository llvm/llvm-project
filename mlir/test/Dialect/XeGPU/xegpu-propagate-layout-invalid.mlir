// RUN: mlir-opt --xegpu-propagate-layout %s | FileCheck %s

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
