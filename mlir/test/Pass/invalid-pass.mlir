// RUN: not mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-module-pass{test-option=a}))' 2>&1 | FileCheck %s

// CHECK: <Pass-Options-Parser>: no such option test-option
// CHECK: failed to add `test-module-pass` with options `test-option=a`
// CHECK: failed to add `builtin.module` with options `` to inner pipeline
module {}
