// RUN: mlir-opt <%s -split-input-file -verify-diagnostics

func @tensor.cast_mismatching_constants(%arg0: tensor<1xf32>) {
  // expected-error@+1 {{operand type 'tensor<1xf32>' and result type 'tensor<2xf32>' are cast incompatible}}
  %0 = tensor.cast %arg0 : tensor<1xf32> to tensor<2xf32>
  return
}

// -----

func @extract_too_many_indices(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = tensor.extract %arg0[] : tensor<?xf32>
  return
}
