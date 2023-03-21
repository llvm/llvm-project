// RUN: mlir-opt %s --log-actions-to=- -canonicalize -test-module-pass | FileCheck %s

// CHECK: [thread {{.*}}] begins (no breakpoint) Action `pass-execution-action` running `Canonicalizer` on Operation `builtin.module` (module {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution-action`
// CHECK-NEXT: [thread {{.*}}] begins (no breakpoint) Action `pass-execution-action`  running `{{.*}}TestModulePass` on Operation `builtin.module` (module {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution-action`
// CHECK-NOT: Action
