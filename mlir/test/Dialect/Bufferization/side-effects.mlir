// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @test_side_effects(%arg0: memref<2xi32>) -> memref<2xi32> {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op result 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
  %0 = bufferization.clone %arg0 : memref<2xi32> to memref<2xi32>
  return %0 : memref<2xi32>
}

// to_buffer without read_only has a write effect on the result (prevents LICM).
func.func @test_to_buffer_write_effect(%arg0: tensor<2xi32>) -> memref<2xi32> {
  // expected-remark @below {{found an instance of 'write' on op result 0, on resource '<Default>'}}
  %0 = bufferization.to_buffer %arg0 : tensor<2xi32> to memref<2xi32>
  return %0 : memref<2xi32>
}

// to_buffer with read_only has no side effects.
func.func @test_to_buffer_readonly_no_effect(%arg0: tensor<2xi32>) -> memref<2xi32> {
  // expected-remark @below {{operation has no memory effects}}
  %0 = bufferization.to_buffer %arg0 read_only : tensor<2xi32> to memref<2xi32>
  return %0 : memref<2xi32>
}
