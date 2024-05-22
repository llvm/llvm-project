// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Verify that ops with broadcastable trait verifies operand and result type
// combinations and emits an error for invalid combinations.

func.func @broadcast_scalar_scalar_scalar(tensor<i32>, tensor<i32>) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

func.func @broadcast_tensor_scalar_tensor(tensor<4xi32>, tensor<i32>) -> tensor<4xi32> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<i32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Check only one dimension has size 1
func.func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check multiple dimensions have size 1
func.func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x5xi32>
  return %0 : tensor<8x7x6x5xi32>
}

// -----

// Check leading unknown dimension
func.func @broadcast_tensor_tensor_tensor(tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32> {
^bb0(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  return %0 : tensor<?x7x6x5xi32>
}

// -----

// Check unknown dimension in the middle
func.func @broadcast_tensor_tensor_tensor(tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32> {
^bb0(%arg0: tensor<8x1x?x1xi32>, %arg1: tensor<7x1x5xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x?x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x?x5xi32>
  return %0 : tensor<8x7x?x5xi32>
}

// -----

// Check incompatible vector and tensor result type
func.func @broadcast_scalar_vector_vector(tensor<4xf32>, tensor<4xf32>) -> vector<4xf32> {
^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
  // expected-error @+1 {{op result #0 must be tensor of any type values, but got 'vector<4xf32>'}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// Check incompatible operand types with known dimension
func.func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x3xi32>):
  // expected-error @+1 {{operands don't have broadcast-compatible shapes}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x3xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// Check incompatible result type with known dimension
func.func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<3x1xi32>):
  // expected-error @+1 {{op result type '4x3x3' not broadcast compatible with broadcasted operands's shapes '4x3x2'}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<3x1xi32>) -> tensor<4x3x3xi32>
  return %0 : tensor<4x3x3xi32>
}

// -----

// Check incompatible result type with known dimension
func.func @broadcast_tensor_tensor_tensor(tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32> {
^bb0(%arg0: tensor<8x1x6x1xi32>, %arg1: tensor<7x1x5xi32>):
  // expected-error @+1 {{op result type '8x7x6x1' not broadcast compatible with broadcasted operands's shapes '8x7x6x5'}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<8x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<8x7x6x1xi32>
  return %0 : tensor<8x7x6x1xi32>
}

// -----

func.func @broadcast_tensor_tensor_tensor(tensor<2xi32>, tensor<2xi32>) -> tensor<*xi32> {
^bb0(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// -----

func.func @broadcast_tensor_tensor_tensor(tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<4x3x2xi32>, %arg1: tensor<?xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4x3x2xi32>, tensor<?xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

// It is alright to have an implicit dynamic-to-static cast in a dimension size
// as long as the runtime result size is consistent with the result tensor's
// static dimension.
func.func @broadcast_tensor_tensor_tensor(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<2xi32> {
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?xi32>, tensor<?xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

func.func @broadcast_tensor_tensor_tensor(%arg0: tensor<?x6x1xi32>, %arg1: tensor<*xi32>) -> tensor<?x6x?xi32> {
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x6x1xi32>, tensor<*xi32>) -> tensor<?x6x?xi32>
  return %0 : tensor<?x6x?xi32>
}

// -----

// Unranked operands but ranked result
func.func @broadcast_tensor_tensor_tensor(tensor<*xi32>, tensor<*xi32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// Unranked operand and compatible ranked result
func.func @broadcast_tensor_tensor_tensor(tensor<3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32> {
^bb0(%arg0: tensor<3x2xi32>, %arg1: tensor<*xi32>):
  %0 = "test.broadcastable"(%arg0, %arg0, %arg1) : (tensor<3x2xi32>, tensor<3x2xi32>, tensor<*xi32>) -> tensor<4x3x2xi32>
  return %0 : tensor<4x3x2xi32>
}

// -----

func.func @broadcast_tensor_tensor_tensor(tensor<3x2xi32>, tensor<*xi32>) -> tensor<2xi32> {
^bb0(%arg0: tensor<3x2xi32>, %arg1: tensor<*xi32>):
  // expected-error @+1 {{op result type '2' not broadcast compatible with broadcasted operands's shapes '3x2'}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<3x2xi32>, tensor<*xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// Correct use of broadcast semantics for input dimensions
func.func @broadcast_tensor_tensor_tensor(%arg0: tensor<?x1x6x1xi32>, %arg1: tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32> {
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<?x1x6x1xi32>, tensor<7x1x5xi32>) -> tensor<?x7x6x5xi32>
  return %0 : tensor<?x7x6x5xi32>
}

// -----

// Incorrect attempt to use broadcast semantics for result
func.func @broadcast_tensor_tensor_tensor(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<5xi32> {
  // expected-error @+1 {{op result type '5' not broadcast compatible with broadcasted operands's shapes '1'}}
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
  return %0 : tensor<5xi32>
}

// -----

func.func @broadcastDifferentResultType(tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1> {
^bb0(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>):
  %0 = "test.broadcastable"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}
