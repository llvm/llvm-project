// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @alloc_tensor_err(%arg0 : index, %arg1 : index)
{
  // expected-error @+1 {{specified type 'tensor<4x?x?x5xf32>' does not match the inferred type 'tensor<4x5x?x?xf32>'}}
  %1 = bufferization.alloc_tensor [4, 5, %arg0, %arg1] : tensor<4x?x?x5xf32>
  return
}

// -----

func.func @alloc_tensor_err(%arg0 : index)
{
  // expected-error @+1 {{expected 4 sizes values}}
  %1 = bufferization.alloc_tensor [4, 5, %arg0] : tensor<4x?x?x5xf32>
  return
}

// -----

func.func @alloc_tensor_err(%arg0 : index)
{
  // expected-error @+1 {{expected 2 dynamic sizes values}}
  %1 = "bufferization.alloc_tensor"(%arg0) {static_sizes = [4, -1, -1, 5]} : (index) -> tensor<4x?x?x5xf32>
  return
}
