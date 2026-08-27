// RUN: mlir-opt %s -verify-diagnostics --affine-super-vectorize=virtual-vector-size=0
// RUN: mlir-opt %s -verify-diagnostics \
// RUN:   -affine-super-vectorize="virtual-vector-size=4,0 vectorize-reductions=true"

// expected-error@+1 {{The 'virtual-vector-size' option must contain only positive values.}}
func.func @with_zero_vector_size(%arg0: memref<21x12x12xi1>) {
  affine.for %arg1 = 0 to 84 step 4294967295 {
  }
  return
}
