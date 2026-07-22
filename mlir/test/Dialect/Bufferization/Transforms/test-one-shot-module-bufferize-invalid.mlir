// RUN: mlir-opt %s -test-one-shot-module-bufferize -verify-diagnostics
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1" -verify-diagnostics

module {
  // expected-error @below {{cannot bufferize function boundary type 'tensor<!llvm.array<1 x i32>>': element type '!llvm.array<1 x i32>' is not a valid memref element type}}
  // expected-error @below {{failed to bufferize op}}
  func.func private @sparse_csr(tensor<f64>) -> tensor<!llvm.array<1 x i32>>
}
