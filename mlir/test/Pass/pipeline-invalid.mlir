// RUN: mlir-opt --no-implicit-module \
// RUN:     --pass-pipeline='any(buffer-deallocation)' --verify-diagnostics \
// RUN:     --split-input-file %s

// Note: "buffer-deallocation" is a function pass. Any other function pass could
// be used for this test.

// expected-error@below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
arith.constant 0

// -----

// expected-error@below {{trying to schedule a pass on an unregistered operation}}
"test.op"() : () -> ()

// -----

// expected-error@below {{trying to schedule a pass on an unsupported operation}}
module {}
