// RUN: mlir-opt %s -verify-diagnostics --affine-super-vectorize

// expected-error@+1 {{Vector rank cannot be zero, specify pass option virtual-vector-size}} 
func.func @with_zero_vector_size(%arg0: memref<21x12x12xi1>) {
  affine.for %arg1 = 0 to 84 step 4294967295 {
  }
  return
}

