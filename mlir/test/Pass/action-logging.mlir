// RUN: mlir-opt %s --log-actions-to=- -canonicalize -test-module-pass | FileCheck %s

// CHECK: [thread {{.*}}] begins (no breakpoint) Action `pass-execution-action` running `Canonicalizer` on Operation `builtin.module`
// CHECK: [thread {{.*}}] completed `pass-execution-action`
// CHECK: [thread {{.*}}] begins (no breakpoint) Action `pass-execution-action` running `(anonymous namespace)::TestModulePass` on Operation `builtin.module`
// CHECK: [thread {{.*}}] completed `pass-execution-action`
