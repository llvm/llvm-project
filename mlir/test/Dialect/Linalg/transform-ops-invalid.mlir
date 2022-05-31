// RUN: mlir-opt %s --split-input-file --verify-diagnostics

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects iterator_interchange to be a permutation, found [1, 1]}}
  transform.structured.interchange %arg0 {iterator_interchange = [1, 1]}
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects padding_dimensions to contain positive integers, found [1, -7]}}
  transform.structured.pad %arg0 {padding_dimensions=[1, -7]}
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects pack_paddings to contain booleans (0/1), found [1, 7]}}
  transform.structured.pad %arg0 {pack_paddings=[1, 7]}
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects hoist_paddings to contain positive integers, found [1, -7]}}
  transform.structured.pad %arg0 {hoist_paddings=[1, -7]}
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error@below {{expects transpose_paddings to be a permutation, found [1, 1]}}
  transform.structured.pad %arg0 {transpose_paddings=[[1, 1]]}
}
