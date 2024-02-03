// RUN: mlir-opt %s --split-input-file --verify-diagnostics

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expected one body argument}}
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1:
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expected body argument to implement TransformHandleTypeInterface}}
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: i32):
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects nested operations to implement MatchOpInterface}}
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-note @below {{offending operation}}
    transform.test_consume_operand %arg1 : !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}
// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects parent op to be 'transform.match.structured'}}
  transform.match.structured.body %arg0 { passthrough } : !transform.any_op
  transform.yield
}


// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{expected predicate to apply to the surrounding structured op}}
    transform.match.structured.body %arg0 { passthrough } : !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{only one of {"reduction_position", "passthrough", "elementwise", "contraction"} is allowed}}
    transform.match.structured.body %arg1 { passthrough, reduction_position = 0 } : !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{cannot request both 'all' and 'inverted' values in the list}}
    "transform.match.structured.dim"(%arg1) { is_all, is_inverted, raw_dim_list = array<i64> } : (!transform.any_op) -> ()
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{cannot both request 'all' and specific values in the list}}
    "transform.match.structured.dim"(%arg1) { is_all, raw_dim_list = array<i64: 0, 1> } : (!transform.any_op) -> ()
    transform.match.structured.yield
  }
  transform.yield
}
// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{must request specific values in the list if 'all' is not specified}}
    "transform.match.structured.dim"(%arg1) { raw_dim_list = array<i64> } : (!transform.any_op) -> ()
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{op expected the listed values to be unique}}
    "transform.match.structured.dim"(%arg1) { raw_dim_list = array<i64: 0, 0> } : (!transform.any_op) -> ()
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):   
    // expected-error @below {{cannot request the same dimension to be both parallel and reduction}}
    "transform.match.structured.dim"(%arg1) { is_all, parallel, reduction, raw_dim_list = array<i64> } : (!transform.any_op) -> ()
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):   
    // expected-error @below {{"permutation" and "projected_permutation" are mutually exclusive}}
    transform.match.structured.input %arg1[all] { permutation, projected_permutation } : !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}
// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):   
    // expected-error @below {{cannot bind multiple inputs/inits to the same value}}
    transform.match.structured.input %arg1[0, 1] : (!transform.any_op) -> !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):   
    // expected-error @below {{"permutation" and "projected_permutation" are mutually exclusive}}
    transform.match.structured.init %arg1[all] { permutation, projected_permutation } : !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}
// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):   
    // expected-error @below {{cannot bind multiple inputs/inits to the same value}}
    transform.match.structured.init %arg1[0, 1] : (!transform.any_op) -> !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{expects either the any/single keyword or the type value handle result type}}
    transform.match.structured.result %arg1[0] : (!transform.any_op) -> !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{expects either the any/single keyword or the type value handle result type}}
    transform.match.structured.result %arg1[0] {any} : (!transform.any_op) -> !transform.any_value
    transform.match.structured.yield
  }
  transform.yield
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  transform.match.structured %arg0 : !transform.any_op {
  ^bb1(%arg1: !transform.any_op):
    // expected-error @below {{'any' and 'single' are mutually exclusive}}
    transform.match.structured.result %arg1[0] {any, single} : (!transform.any_op) -> !transform.any_op
    transform.match.structured.yield
  }
  transform.yield
}
