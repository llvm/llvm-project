// UNSUPPORTED: system-windows
// RUN: not mlir-reduce -reduction-tree --no-implicit-module %s 2>&1 | FileCheck %s --check-prefix=CHECK-TREE

// The reduction passes are currently restricted to 'builtin.module'.
// CHECK-TREE: error: top-level op must be 'builtin.module'
func.func private @foo()
