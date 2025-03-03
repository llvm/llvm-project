// RUN: mlir-opt --no-implicit-module \
// RUN:     --pass-pipeline='any(test-function-pass)' --verify-diagnostics \
// RUN:     --split-input-file %s

// Note: "test-function-pass" is a function pass. Any other function pass could
// be used for this test.

// expected-error@below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
arith.constant 0

// -----

// expected-error@below {{trying to schedule a pass on an unregistered operation}}
"test.op"() : () -> ()

// -----

// expected-error@below {{trying to schedule a pass on an unsupported operation}}
module {}
