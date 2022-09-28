// RUN: mlir-opt --no-implicit-module --canonicalize --verify-diagnostics --split-input-file

// expected-error@below {{trying to schedule a pass on an operation not marked as 'IsolatedFromAbove'}}
arith.constant 0

// -----

// expected-error@below {{trying to schedule a pass on an unregistered operation}}
"test.op"() : () -> ()
