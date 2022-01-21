// RUN: mlir-opt -canonicalize %s | FileCheck %s

pdl.pattern @operation_op : benefit(1) {
  %root = pdl.operation "foo.op"
  pdl.rewrite %root {
    // CHECK: pdl.operation "bar.unused"
    %unused_rewrite = pdl.operation "bar.unused"
    pdl.erase %root
  }
}
