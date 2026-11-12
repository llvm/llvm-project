// RUN: mlir-opt --xegpu-propagate-layout --split-input-file -verify-diagnostics %s | FileCheck %s

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

// Regression test for https://github.com/llvm/llvm-project/issues/221142:
// --xegpu-propagate-layout must not crash on a module without any XeGPU ops.
// assignResultLayout created a xegpu.convert_layout for the scalar result of
// a vector.reduction even though no layout existed anywhere in the module;
// with the XeGPU dialect never loaded (no XeGPU ops and no dependentDialects
// entry), the op creation aborted with "Property type mismatch: TypeID does
// not match requested type". The pass now bails out when the result carries
// no layout and reports the missing layout via diagnostics instead.
module {
  func.func @reduction(%arg0: f32) -> () {
    // expected-warning@+1 {{op has users but no layout assigned for its result}}
    %cst_1 = arith.constant dense<1.000000e+00> : vector<16xf32>
    %0 = vector.reduction <add>, %cst_1 : vector<16xf32> into f32
    return
  }
}
