// RUN: mlir-opt %s -test-pass-state-extension-communication -verify-diagnostics | FileCheck %s

// CHECK: Printing opCollection before processing transform ops, size: 1
// CHECK: PASS-TRANSFORMOP-PASS

// CHECK: Printing opCollection after processing transform ops, size: 4
// CHECK: PASS-TRANSFORMOP-PASS transform.test_initializer_extension_A transform.test_initializer_extension_B transform.test_initializer_extension_C

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{Number of currently registered op: 1}}
    transform.test_initializer_extension "A"
    // expected-remark @below {{Number of currently registered op: 2}}
    transform.test_initializer_extension "B"
    // expected-remark @below {{Number of currently registered op: 3}}
    transform.test_initializer_extension "C"
    transform.yield
  }
}
