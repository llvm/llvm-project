// RUN: mlir-opt %s -test-loop-fusion=test-fusion-compute-cost \
// RUN:   -verify-diagnostics

func.func @single_iteration_store_load(%input: memref<1xf32>, %value: f32) {
  affine.for %i = 0 to 1 {
    affine.store %value, %input[%i] : memref<1xf32>
  }
  // expected-remark@below {{fusion compute cost: 1}}
  affine.for %j = 0 to 1 {
    affine.for %k = 0 to 1 {
      %loaded = affine.load %input[%k] : memref<1xf32>
      %unused = arith.addf %loaded, %loaded : f32
    }
  }
  return
}
