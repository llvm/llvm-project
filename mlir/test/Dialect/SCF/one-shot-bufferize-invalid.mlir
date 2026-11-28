// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops allow-return-allocs-from-loops" -split-input-file -verify-diagnostics

func.func @inconsistent_memory_space_scf_if(%c: i1) -> tensor<10xf32> {
  // Yielding tensors with different memory spaces. Such IR cannot be
  // bufferized.
  %0 = bufferization.alloc_tensor() <{memory_space = 0 : ui64}> : tensor<10xf32>
  %1 = bufferization.alloc_tensor() <{memory_space = 1 : ui64}> : tensor<10xf32>
  // expected-error @+1 {{incompatible buffer types on then/else branches}}
  %r = scf.if %c -> tensor<10xf32> {
    // expected-error @+1 {{failed to bufferize op}}
    scf.yield %0 : tensor<10xf32>
  } else {
    scf.yield %1 : tensor<10xf32>
  }
  func.return %r : tensor<10xf32>
}

// -----

func.func @execute_region_multiple_yields(%t: tensor<5xf32>) -> tensor<5xf32> {
  // expected-error @below{{op op without unique scf.yield is not supported}}
  %0 = scf.execute_region -> tensor<5xf32> {
    scf.yield %t : tensor<5xf32>
  ^bb1(%arg1 : tensor<5xf32>):
    scf.yield %arg1 : tensor<5xf32>
  }
  func.return %0 : tensor<5xf32>
}

// -----

func.func @execute_region_no_yield(%t: tensor<5xf32>) -> tensor<5xf32> {
  // expected-error @below{{op op without unique scf.yield is not supported}}
  %0 = scf.execute_region -> tensor<5xf32> {
    cf.br ^bb0(%t : tensor<5xf32>)
  ^bb0(%arg0 : tensor<5xf32>):
    cf.br ^bb1(%arg0: tensor<5xf32>)
  ^bb1(%arg1 : tensor<5xf32>):
    cf.br ^bb0(%arg1: tensor<5xf32>)
  }
  func.return %0 : tensor<5xf32>
}

// -----

func.func @inconsistent_memory_space_scf_for(%lb: index, %ub: index, %step: index) -> tensor<10xf32> {
  %0 = bufferization.alloc_tensor() <{memory_space = 0 : ui64}> : tensor<10xf32>
  %1 = bufferization.alloc_tensor() <{memory_space = 1 : ui64}> : tensor<10xf32>
  // expected-error @below{{init_arg and yielded value bufferize to incompatible buffer types}}
  %2 = scf.for %iv = %lb to %ub step %step iter_args(%arg = %0) -> tensor<10xf32> {
    // expected-error @below {{failed to bufferize op}}
    scf.yield %1 : tensor<10xf32>
  }
  return %2 : tensor<10xf32>
}

// -----

// The two branches' buffers have layouts the default cast cannot cast to the
// type they are reconciled to: the yield is not bufferized.

func.func @if_cast_refused(
    %c: i1,
    // expected-error @+1 {{cannot cast a buffer of type 'memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>>' to 'memref<4xf32, strided<[?], offset: ?>>'}}
    %a: memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>>,
    %b: memref<4xf32>) -> f32 {
  %ta = bufferization.to_tensor %a restrict
    : memref<4xf32, affine_map<(d0) -> (d0 floordiv 2)>> to tensor<4xf32>
  %tb = bufferization.to_tensor %b restrict : memref<4xf32> to tensor<4xf32>
  %s = scf.if %c -> (tensor<4xf32>) {
    // expected-error @+1 {{failed to bufferize op}}
    scf.yield %ta : tensor<4xf32>
  } else {
    scf.yield %tb : tensor<4xf32>
  }
  %c0 = arith.constant 0 : index
  %e = tensor.extract %s[%c0] : tensor<4xf32>
  return %e : f32
}
