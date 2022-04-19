// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{expected the entry block to have one argument of type '!pdl.operation'}}
transform.sequence {
}

// -----

// expected-note @below {{nested in another sequence}}
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expected the root operation to be provided for a nested sequence}}
  transform.sequence {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expected children ops to implement TransformOpInterface}}
transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{op without interface}}
  arith.constant 42.0 : f32
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one use}}
  %0 = transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
  } : !pdl.operation
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 {
  ^bb2(%arg2: !pdl.operation):
  }
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 {
  ^bb3(%arg3: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects the types of the terminator operands to match the types of the resul}}
%0 = transform.sequence {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{terminator}}
  transform.yield
} : !pdl.operation
