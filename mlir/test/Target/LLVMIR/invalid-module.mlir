// RUN: mlir-translate -verify-diagnostics -mlir-to-llvmir --no-implicit-module %s

// expected-error@below {{'llvm.func' op can not be translated to an LLVMIR module}}
llvm.func @foo() {
  llvm.return
}
