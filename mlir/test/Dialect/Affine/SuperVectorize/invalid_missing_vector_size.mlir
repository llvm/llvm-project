// RUN: mlir-opt %s -verify-diagnostics --affine-super-vectorize

// expected-error@+1 {{The 'virtual-vector-size' option must be specified.}}
func.func @missing_vector_size() {
  return
}
