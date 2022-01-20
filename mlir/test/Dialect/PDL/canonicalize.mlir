// RUN: mlir-opt -canonicalize %s | FileCheck %s

pdl.pattern @operation_op : benefit(1) {
  %root = operation "foo.op"
  rewrite %root {
    // CHECK: operation "bar.unused"
    %unused_rewrite = operation "bar.unused"
    erase %root
  }
}
