// UNSUPPORTED: system-windows
// RUN: not mlir-reduce -opt-reduction-pass --no-implicit-module %s |& FileCheck %s --check-prefix=CHECK-PASS
// RUN: not mlir-reduce -reduction-tree --no-implicit-module %s |& FileCheck %s --check-prefix=CHECK-TREE

// The reduction passes are currently restricted to 'builtin.module'.
// CHECK-PASS: error: Can't add pass '{{.+}}' restricted to 'builtin.module' on a PassManager intended to run on 'func.func'
// CHECK-TREE: error: top-level op must be 'builtin.module'
func.func private @foo()
