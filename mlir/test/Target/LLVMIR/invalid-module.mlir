// RUN: mlir-translate -verify-diagnostics -mlir-to-llvmir --no-implicit-module -split-input-file %s

// expected-error@below {{'llvm.func' op can not be translated to an LLVMIR module}}
llvm.func @foo() {
  llvm.return
}

// -----

// expected-error@below {{expected an array attribute for a module level asm}}
module attributes {llvm.module_asm = "foo"} {}

// -----

// expected-error@below {{expected a string attribute for each entry of a module level asm}}
module attributes {llvm.module_asm = [42]} {}
