// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @reduce_op_args_num_mismatch(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{ReduceOp body is expected to have 3 arguments}}
  %num_elements = shape.reduce(%shape, %init) -> !shape.size {
    ^bb0(%index: index, %dim: !shape.size):
      "shape.yield"(%dim) : (!shape.size) -> ()
  }
}

// -----

func @reduce_op_arg0_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 0 of ReduceOp body is expected to be of IndexType}}
  %num_elements = shape.reduce(%shape, %init) -> !shape.size {
    ^bb0(%index: f32, %dim: !shape.size, %lci: !shape.size):
      %acc = "shape.add"(%lci, %dim) : (!shape.size, !shape.size) -> !shape.size
      "shape.yield"(%acc) : (!shape.size) -> ()
  }
}

// -----

func @reduce_op_arg1_wrong_type(%shape : !shape.shape, %init : !shape.size) {
  // expected-error@+1 {{argument 1 of ReduceOp body is expected to be of SizeType}}
  %num_elements = shape.reduce(%shape, %init) -> !shape.size {
    ^bb0(%index: index, %dim: f32, %lci: !shape.size):
      "shape.yield"() : () -> ()
  }
}

// -----

func @reduce_op_init_type_mismatch(%shape : !shape.shape, %init : f32) {
  // expected-error@+1 {{type mismatch between argument 2 of ReduceOp body and initial value 0}}
  %num_elements = shape.reduce(%shape, %init) -> f32 {
    ^bb0(%index: index, %dim: !shape.size, %lci: !shape.size):
      "shape.yield"() : () -> ()
  }
}
