// RUN: not mlir-opt %s --remarks-filter="category-1-passed" --remark-policy=final \
// RUN:   --pass-pipeline='builtin.module(test-remark,test-pass-failure)' 2>&1 | FileCheck %s

// The failing pipeline returns before the textual output; the remark deferred
// by the final policy must still be printed.

// CHECK: remark: [Passed] test-remark | Category:category-1-passed
"test.op"() : () -> ()
