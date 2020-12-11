// RUN: mlir-opt <%s -verify-diagnostics

// -----

func @extract_too_many_indices(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{incorrect number of indices for extract_element}}
  %0 = tensor.extract %arg0[] : tensor<?xf32>
  return
}
