// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @test_side_effects(%arg0: memref<8xf32>) {
  // expected-remark @below {{operation has no memory effects}}
  %c0 = arith.constant 0 : index
  // expected-remark @below {{operation has no memory effects}}
  %c4 = arith.constant 4 : index
  // expected-remark @below {{operation has no memory effects}}
  %cst = arith.constant 0.0 : f32
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<8xf32>, vector<4xf32>
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  vector.transfer_write %0, %arg0[%c4] : vector<4xf32>, memref<8xf32>
  return
}
