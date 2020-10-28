// RUN: mlir-opt -split-input-file %s -verify-diagnostics

// CHECK-LABEL: test_index_cast_shape_error
func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{requires the same shape for all operands and results}}
  %0 = index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

// CHECK-LABEL: test_index_cast_tensor_error
func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{requires the same shape for all operands and results}}
  %0 = index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}

// -----

func @dynamic_tensor_from_elements(%m : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have as many index operands as dynamic extents in the result type}}
  %tnsr = dynamic_tensor_from_elements %m {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{must have one body argument per input dimension}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{all body arguments must be index}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : i64):
      %elem = constant 8.0 : f32
      yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+2 {{op expects regions to end with 'std.yield', found 'std.return'}}
  // expected-note @+1 {{in custom textual format, the absence of terminator implies 'std.yield'}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8.0 : f32
      return %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// -----

func @dynamic_tensor_from_elements(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  // expected-error @+1 {{body must be terminated with a `yield` operation of the tensor element type}}
  %tnsr = dynamic_tensor_from_elements %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = constant 8 : i32
      yield %elem : i32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}
