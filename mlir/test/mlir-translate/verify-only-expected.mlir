// Check that verify-diagnostics=only-expected passes with only one actual `expected-error`
// RUN: mlir-translate %s --allow-unregistered-dialect -verify-diagnostics=only-expected -split-input-file -mlir-to-llvmir

// Check that verify-diagnostics=all fails because we're missing two `expected-error`
// RUN: not mlir-translate %s --allow-unregistered-dialect -verify-diagnostics=all -split-input-file -mlir-to-llvmir 2>&1 | FileCheck %s --check-prefix=CHECK-VERIFY-ALL
// CHECK-VERIFY-ALL: unexpected error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: simple.terminator1
// CHECK-VERIFY-ALL: unexpected error: cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: simple.terminator3

llvm.func @trivial() {
  "simple.terminator1"() : () -> ()
}

// -----

llvm.func @trivial() {
  // expected-error @+1 {{cannot be converted to LLVM IR: missing `LLVMTranslationDialectInterface` registration for dialect for op: simple.terminator2}}
  "simple.terminator2"() : () -> ()
}

// -----

llvm.func @trivial() {
  "simple.terminator3"() : () -> ()
}
