// RUN: mlir-opt %s -pass-pipeline='builtin.module(gpu.module(select-pass{ \
// RUN:     name=TestSelectPass \
// RUN:     select-cond-name=test.attr \
// RUN:     select-values=rocdl,nvvm \
// RUN:     select-pipelines=convert-gpu-to-rocdl,convert-gpu-to-nvvm \
// RUN:     }))' | FileCheck %s

gpu.module @rocdl_module attributes {test.attr = "rocdl"} {
// CHECK-LABEL: func @foo()
// CHECK: rocdl.workitem.id.x
  func.func @foo() -> index {
    %0 = gpu.thread_id x
    return %0 : index
  }
}

gpu.module @nvvm_module attributes {test.attr = "nvvm"} {
// CHECK-LABEL: func @bar()
// CHECK: nvvm.read.ptx.sreg.tid.x
  func.func @bar() -> index {
    %0 = gpu.thread_id x
    return %0 : index
  }
}
