// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops allow-return-allocs" -split-input-file -verify-diagnostics

func.func @inconsistent_memory_space_scf_if(%c: i1) -> tensor<10xf32> {
  // Yielding tensors with different memory spaces. Such IR cannot be
  // bufferized.
  %0 = bufferization.alloc_tensor() {memory_space = 0 : ui64} : tensor<10xf32>
  %1 = bufferization.alloc_tensor() {memory_space = 1 : ui64} : tensor<10xf32>
  // expected-error @+1 {{inconsistent memory space on then/else branches}}
  %r = scf.if %c -> tensor<10xf32> {
    // expected-error @+1 {{failed to bufferize op}}
    scf.yield %0 : tensor<10xf32>
  } else {
    scf.yield %1 : tensor<10xf32>
  }
  func.return %r : tensor<10xf32>
}

// -----

func.func @execute_region_multiple_blocks(%t: tensor<5xf32>) -> tensor<5xf32> {
  // expected-error @below{{op or BufferizableOpInterface implementation does not support unstructured control flow, but at least one region has multiple blocks}}
  %0 = scf.execute_region -> tensor<5xf32> {
    cf.br ^bb1(%t : tensor<5xf32>)
  ^bb1(%arg1 : tensor<5xf32>):
    scf.yield %arg1 : tensor<5xf32>
  }
  func.return %0 : tensor<5xf32>
}

// -----

func.func @inconsistent_memory_space_scf_for(%lb: index, %ub: index, %step: index) -> tensor<10xf32> {
  %0 = bufferization.alloc_tensor() {memory_space = 0 : ui64} : tensor<10xf32>
  %1 = bufferization.alloc_tensor() {memory_space = 1 : ui64} : tensor<10xf32>
  // expected-error @below{{init_arg and yielded value bufferize to inconsistent memory spaces}}
  %2 = scf.for %iv = %lb to %ub step %step iter_args(%arg = %0) -> tensor<10xf32> {
    // expected-error @below {{failed to bufferize op}}
    scf.yield %1 : tensor<10xf32>
  }
  return %2 : tensor<10xf32>
}
