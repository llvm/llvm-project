// RUN: mlir-opt %s --verify-diagnostics --remarks-filter="category-1-passed" \
// RUN:   --remark-policy=final \
// RUN:   --pass-pipeline='builtin.module(test-remark,test-pass-failure)'

// The failing pipeline returns before the textual output; the remark deferred
// by the final policy must still reach the diagnostic verifier.
module {
  // expected-remark@below {{[Passed] test-remark | Category:category-1-passed}}
  "test.op"() : () -> ()
}
