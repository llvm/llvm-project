// RUN: mlir-opt %s --log-actions-to=- -test-stats-pass -test-module-pass | FileCheck %s

// CHECK: [thread {{.*}}] begins (no breakpoint) Action `pass-execution` running `{{.*}}TestStatisticPass` on Operation `builtin.module` (module {...})`
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-NEXT: [thread {{.*}}] begins (no breakpoint) Action `pass-execution`  running `{{.*}}TestModulePass` on Operation `builtin.module` (module {...}
// CHECK-NEXT: [thread {{.*}}] completed `pass-execution`
// CHECK-NOT: Action
