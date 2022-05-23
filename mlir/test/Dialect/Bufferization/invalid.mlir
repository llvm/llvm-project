// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @alloc_tensor_err(%arg0 : index)
{
  // expected-error @+1 {{expected 2 dynamic sizes}}
  %1 = bufferization.alloc_tensor(%arg0) : tensor<4x?x?x5xf32>
  return
}
