// RUN: not mlir-opt %s -pass-pipeline='builtin.module(builtin.module(test-module-pass{test-option=a}))' 2>&1 | FileCheck %s
// RUN: not mlir-opt %s -mlir-print-ir-module-scope -mlir-print-ir-before=cse 2>&1 | FileCheck -check-prefix=PRINT_MODULE_IR_WITH_MULTITHREAD %s

// CHECK: <Pass-Options-Parser>: no such option test-option
// CHECK: failed to add `test-module-pass` with options `test-option=a`
// CHECK: failed to add `builtin.module` with options `` to inner pipeline
module {}

// PRINT_MODULE_IR_WITH_MULTITHREAD: IR print for module scope can't be setup on a pass-manager without disabling multi-threading first.
