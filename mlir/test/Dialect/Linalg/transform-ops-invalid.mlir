// RUN: mlir-opt %s --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{'transform.structured.interchange' op expects iterator_interchange to be a permutation, found 1, 1}}
  transform.structured.interchange %arg0 iterator_interchange = [1, 1] 
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects padding_dimensions to contain positive integers, found [1, -7]}}
  transform.structured.pad %arg0 {padding_dimensions=[1, -7]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects pack_paddings to contain booleans (0/1), found [1, 7]}}
  transform.structured.pad %arg0 {pack_paddings=[1, 7]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects hoist_paddings to contain positive integers, found [1, -7]}}
  transform.structured.pad %arg0 {hoist_paddings=[1, -7]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects transpose_paddings to be a permutation, found [1, 1]}}
  transform.structured.pad %arg0 {transpose_paddings=[[1, 1]]}
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{'transform.structured.interchange' op attribute 'iterator_interchange' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  transform.structured.interchange %arg0 iterator_interchange = [-3, 1]
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects all results type to be the same}}
  "transform.structured.multitile_sizes"(%arg0) { target_size = 3, divisor = 2, dimension = 0 }
      : (!pdl.operation) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i32>)
}
