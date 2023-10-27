// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @test_side_effects(%arg0: memref<2xi32>) -> memref<2xi32> {
  // expected-remark @below {{found an instance of 'read' on a value, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on a value, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'allocate' on a value, on resource '<Default>'}}
  %0 = bufferization.clone %arg0 : memref<2xi32> to memref<2xi32>
  // expected-remark @below {{operation has no memory effects}}
  return %0 : memref<2xi32>
}
