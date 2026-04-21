// UNSUPPORTED: system-windows
// RUN: not mlir-reduce -reduction-tree --no-implicit-module %s 2>&1 | FileCheck %s --check-prefix=CHECK-TREE
// RUN: not mlir-reduce -reduction-tree='traversal-mode=0 test=%S/false.sh' %s 2>&1 | FileCheck %s --check-prefix=CHECK-INTERESTING

// The reduction passes are currently restricted to 'builtin.module'.
//        CHECK-TREE: error: top-level op must be 'builtin.module'

// CHECK-INTERESTING: error: uninterested module will not be reduced
func.func private @foo()
