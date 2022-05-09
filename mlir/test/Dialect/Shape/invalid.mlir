// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @reduce_op_args_num_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{ReduceOp body is expected to have 3 arguments}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size):
      shape.yield %dim : !shape.size
  }
  return
}

// -----

func.func @reduce_op_arg0_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 0 of ReduceOp body is expected to be of IndexType}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: f32, %dim: !shape.size, %acc: !shape.size):
      %new_acc = "shape.add"(%acc, %dim)
          : (!shape.size, !shape.size) -> !shape.size
      shape.yield %new_acc : !shape.size
  }
  return
}

// -----

func.func @reduce_op_arg1_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 1 of ReduceOp body is expected to be of SizeType if the ReduceOp operates on a ShapeType}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: f32, %lci: !shape.size):
      shape.yield
  }
  return
}

// -----

func.func @reduce_op_arg1_wrong_type(%shape : tensor<?xindex>, %init : index) {
  // expected-error@+1 {{argument 1 of ReduceOp body is expected to be of IndexType if the ReduceOp operates on an extent tensor}}
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index: index, %dim: f32, %lci: index):
      shape.yield
  }
  return
}

// -----

func.func @reduce_op_init_type_mismatch(%shape : !shape.shape, %init : f32) {
  // expected-error@+1 {{type mismatch between argument 2 of ReduceOp body and initial value 0}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> f32 {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      shape.yield
  }
  return
}

// -----

func.func @yield_op_args_num_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+3 {{number of operands does not match number of results of its parent}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      shape.yield %dim, %dim : !shape.size, !shape.size
  }
  return
}

// -----

func.func @yield_op_type_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+4 {{types mismatch between yield op and its parent}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      %c0 = arith.constant 1 : index
      shape.yield %c0 : index
  }
  return
}

// -----

func.func @assuming_all_op_too_few_operands() {
  // expected-error@+1 {{no operands specified}}
  %w0 = shape.assuming_all
  return
}

// -----

func.func @shape_of(%value_arg : !shape.value_shape,
               %shaped_arg : tensor<?x3x4xf32>) {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `shape` to propagate them}}
  %0 = shape.shape_of %value_arg : !shape.value_shape -> tensor<?xindex>
  return
}

// -----

func.func @shape_of_incompatible_return_types(%value_arg : tensor<1x2xindex>) {
  // expected-error@+1 {{'shape.shape_of' op inferred type(s) 'tensor<2xindex>' are incompatible with return type(s) of operation 'tensor<3xindex>'}}
  %0 = shape.shape_of %value_arg : tensor<1x2xindex> -> tensor<3xindex>
  return
}

// -----

func.func @rank(%arg : !shape.shape) {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `size` to propagate them}}
  %0 = shape.rank %arg : !shape.shape -> index
  return
}

// -----

func.func @get_extent(%arg : tensor<?xindex>) -> index {
  %c0 = shape.const_size 0
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `size` to propagate them}}
  %result = shape.get_extent %arg, %c0 : tensor<?xindex>, !shape.size -> index
  return %result : index
}

// -----

func.func @mul(%lhs : !shape.size, %rhs : index) -> index {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `size` to propagate them}}
  %result = shape.mul %lhs, %rhs : !shape.size, index -> index
  return %result : index
}

// -----

func.func @num_elements(%arg : !shape.shape) -> index {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `size` to propagate them}}
  %result = shape.num_elements %arg : !shape.shape -> index
  return %result : index
}

// -----

func.func @add(%lhs : !shape.size, %rhs : index) -> index {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `size` to propagate them}}
  %result = shape.add %lhs, %rhs : !shape.size, index -> index
  return %result : index
}

// -----

func.func @broadcast(%arg0 : !shape.shape, %arg1 : !shape.shape) -> tensor<?xindex> {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `shape` to propagate them}}
  %result = shape.broadcast %arg0, %arg1
      : !shape.shape, !shape.shape -> tensor<?xindex>
  return %result : tensor<?xindex>
}


// -----

func.func @broadcast(%arg0 : !shape.shape, %arg1 : tensor<?xindex>) -> tensor<?xindex> {
  // expected-error@+1 {{if at least one of the operands can hold error values then the result must be of type `shape` to propagate them}}
  %result = shape.broadcast %arg0, %arg1
      : !shape.shape, tensor<?xindex> -> tensor<?xindex>
  return %result : tensor<?xindex>
}

// -----

// Test using an unsupported shape.lib attribute type.

// expected-error@+1 {{only SymbolRefAttr allowed in shape.lib attribute array}}
module attributes {shape.lib = [@shape_lib, "shape_lib"]} {

shape.function_library @shape_lib {
  // Test shape function that returns the shape of input arg as result shape.
  func.func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
    %0 = shape.shape_of %arg : !shape.value_shape -> !shape.shape
    return %0 : !shape.shape
  }
} mapping {
  test.same_operand_result_type = @same_result_shape
}

}

// -----

// Test that duplicate op to shape function mappings are flagged, this uses
// the same library twice for easy overlap.

// expected-error@+1 {{only one op to shape mapping allowed}}
module attributes {shape.lib = [@shape_lib, @shape_lib]} {

shape.function_library @shape_lib {
  // Test shape function that returns the shape of input arg as result shape.
  func.func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
    %0 = shape.shape_of %arg : !shape.value_shape -> !shape.shape
    return %0 : !shape.shape
  }
} mapping {
  test.same_operand_result_type = @same_result_shape
}

}

// -----

// Test that duplicate op to shape function mappings are flagged (this is
// more an invariant of using the dictionary attribute here than anything
// specific to function library op).

module attributes {shape.lib = [@shape_lib]} {

shape.function_library @shape_lib {
  // Test shape function that returns the shape of input arg as result shape.
  func.func @same_result_shape(%arg: !shape.value_shape) -> !shape.shape {
    %0 = shape.shape_of %arg : !shape.value_shape -> !shape.shape
    return %0 : !shape.shape
  }
} mapping {
  // expected-error @+2 {{duplicate key}}
  test.same_operand_result_type = @same_result_shape,
  test.same_operand_result_type = @same_result_shape
}

}

// -----

// Test that op referred to by shape lib is a shape function library.

// expected-error@+1 {{required to be shape function library}}
module attributes {shape.lib = @fn} {

func.func @fn(%arg: !shape.value_shape) -> !shape.shape {
  %0 = shape.shape_of %arg : !shape.value_shape -> !shape.shape
  return %0 : !shape.shape
}

}

// -----

// Test that op referred to by shape lib is a shape function library.

func.func @fn(%arg: !shape.value_shape) -> !shape.shape {
  // expected-error@+1 {{SymbolTable}}
  %0 = shape.shape_of %arg {shape.lib = @fn} : !shape.value_shape -> !shape.shape
  return %0 : !shape.shape
}

// -----

// Test that shape function library is defined.

// expected-error@+1 {{@fn not found}}
module attributes {shape.lib = @fn} { }

// -----

func.func @fn(%arg: !shape.shape) -> !shape.witness {
  // expected-error@+1 {{required at least 2 input shapes}}
  %0 = shape.cstr_broadcastable %arg : !shape.shape
  return %0 : !shape.witness
}

// -----

// Test that type inference flags the wrong return type.

func.func @const_shape() {
  // expected-error@+1 {{'tensor<3xindex>' are incompatible with return type(s) of operation 'tensor<2xindex>'}}
  %0 = shape.const_shape [4, 5, 6] : tensor<2xindex>
  return
}
