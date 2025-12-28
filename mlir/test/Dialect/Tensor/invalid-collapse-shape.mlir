// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// This test checks that an empty reassociation group in `tensor.collapse_shape`
// results in a proper error instead of an assert/crash.

// -----

func.func @test_empty_reassociation(%arg0: tensor<1x?xf32>) -> tensor<?x10xf32> {
  // expected-error@+1 {{'tensor.collapse_shape' op reassociation indices must not be empty}}
  %0 = tensor.collapse_shape %arg0 [[0, 1], []] : tensor<1x?xf32> into tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

