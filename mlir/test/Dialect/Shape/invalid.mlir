// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @reduce_op_args_num_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{ReduceOp body is expected to have 3 arguments}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size):
      shape.yield %dim : !shape.size
  }
}

// -----

func @reduce_op_arg0_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 0 of ReduceOp body is expected to be of IndexType}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: f32, %dim: !shape.size, %acc: !shape.size):
      %new_acc = "shape.add"(%acc, %dim)
          : (!shape.size, !shape.size) -> !shape.size
      shape.yield %new_acc : !shape.size
  }
}

// -----

func @reduce_op_arg1_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 1 of ReduceOp body is expected to be of SizeType if the ReduceOp operates on a ShapeType}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: f32, %lci: !shape.size):
      shape.yield
  }
}

// -----

func @reduce_op_arg1_wrong_type(%shape : tensor<?xindex>, %init : index) {
  // expected-error@+1 {{argument 1 of ReduceOp body is expected to be of IndexType if the ReduceOp operates on an extent tensor}}
  %num_elements = shape.reduce(%shape, %init) : tensor<?xindex> -> index {
    ^bb0(%index: index, %dim: f32, %lci: index):
      shape.yield
  }
}

// -----

func @reduce_op_init_type_mismatch(%shape : !shape.shape, %init : f32) {
  // expected-error@+1 {{type mismatch between argument 2 of ReduceOp body and initial value 0}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> f32 {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      shape.yield
  }
}

// -----

func @yield_op_args_num_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+3 {{number of operands does not match number of results of its parent}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      shape.yield %dim, %dim : !shape.size, !shape.size
  }
}

// -----

func @yield_op_type_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+4 {{types mismatch between yield op and its parent}}
  %num_elements = shape.reduce(%shape, %init) : !shape.shape -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      %c0 = constant 1 : index
      shape.yield %c0 : index
  }
}

// -----

func @assuming_all_op_too_few_operands() {
  // expected-error@+1 {{no operands specified}}
  %w0 = shape.assuming_all
  return
}
