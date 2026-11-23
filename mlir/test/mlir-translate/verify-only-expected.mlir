// Check that verify-diagnostics=only-expected passes with only one actual `expected-error`
// RUN: mlir-translate %s -verify-diagnostics=only-expected -split-input-file -mlir-to-llvmir

// Check that verify-diagnostics=all fails because we're missing two `expected-error`
// RUN: not mlir-translate %s -verify-diagnostics=all -split-input-file -mlir-to-llvmir 2>&1 | FileCheck %s --check-prefix=CHECK-VERIFY-ALL
// CHECK-VERIFY-ALL: unexpected error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: spirv.Undef
// CHECK-VERIFY-ALL: unexpected error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: spirv.Undef

llvm.func @trivial() {
  %0 = spirv.Undef : i32
  llvm.return
}

// -----

llvm.func @trivial() {
  // expected-error @+1 {{cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: spirv.Undef}}
  %0 = spirv.Undef : i32
  llvm.return
}

// -----

llvm.func @trivial() {
  %0 = spirv.Undef : i32
  llvm.return
}
