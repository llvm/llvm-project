// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @alloc_tensor_missing_dims(%arg0: index)
{
  // expected-error @+1 {{expected 2 dynamic sizes}}
  %0 = bufferization.alloc_tensor(%arg0) : tensor<4x?x?x5xf32>
  return
}

// -----

// expected-note @+1 {{prior use here}}
func.func @alloc_tensor_type_mismatch(%t: tensor<?xf32>) {
  // expected-error @+1{{expects different type than prior uses: 'tensor<5xf32>' vs 'tensor<?xf32>'}}
  %0 = bufferization.alloc_tensor() copy(%t) : tensor<5xf32>
  return
}

// -----

func.func @alloc_tensor_copy_and_dims(%t: tensor<?xf32>, %sz: index) {
  // expected-error @+1{{dynamic sizes not needed when copying a tensor}}
  %0 = bufferization.alloc_tensor(%sz) copy(%t) : tensor<?xf32>
  return
}

// -----

func.func @alloc_tensor_invalid_escape_attr(%sz: index) {
  // expected-error @+1{{op attribute 'escape' failed to satisfy constraint: bool attribute}}
  %0 = bufferization.alloc_tensor(%sz) {escape = 5} : tensor<?xf32>
  return
}
