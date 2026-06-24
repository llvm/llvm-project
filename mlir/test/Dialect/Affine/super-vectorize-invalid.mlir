// RUN: mlir-opt %s -affine-super-vectorize -verify-diagnostics

func.func @test_no_vector_size() { // expected-error {{virtual-vector-size option must be specified}}
  return
}
