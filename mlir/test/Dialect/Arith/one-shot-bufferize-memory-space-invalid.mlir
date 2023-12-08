// RUN: mlir-opt %s -one-shot-bufferize="must-infer-memory-space" -split-input-file -verify-diagnostics

func.func @inconsistent_memory_space_arith_select(%c: i1) -> tensor<10xf32> {
  // Selecting tensors with different memory spaces. Such IR cannot be
  // bufferized.
  %0 = bufferization.alloc_tensor() {memory_space = 0 : ui64} : tensor<10xf32>
  %1 = bufferization.alloc_tensor() {memory_space = 1 : ui64} : tensor<10xf32>
  // expected-error @+2 {{inconsistent memory space on true/false operands}}
  // expected-error @+1 {{failed to bufferize op}}
  %r = arith.select %c, %0, %1 : tensor<10xf32>
  func.return %r : tensor<10xf32>
}

// -----

func.func @unknown_memory_space(%idx: index, %v: i32) -> tensor<3xi32> {
  // expected-error @+2 {{could not infer memory space}}
  // expected-error @+1 {{failed to bufferize op}}
  %cst = arith.constant dense<[5, 1000, 20]> : tensor<3xi32>
  %0 = tensor.insert %v into %cst[%idx] : tensor<3xi32>
  return %0 : tensor<3xi32>
}