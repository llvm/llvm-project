// RUN: mlir-opt --test-emulate-narrow-int --verify-diagnostics %s

func.func @memref_load_vector_element(%arg0: memref<4xvector<1xi1>>, %idx: index) {
  // expected-error @+1 {{failed to legalize operation 'memref.load'}}
  %0 = memref.load %arg0[%idx] : memref<4xvector<1xi1>>
  return
}
