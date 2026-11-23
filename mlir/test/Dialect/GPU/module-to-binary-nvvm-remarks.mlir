// REQUIRES: host-supports-nvptx
// RUN: mlir-opt %s --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' --remarks-filter-passed=llvm-inline 2>&1 | FileCheck %s
// RUN: mlir-opt %s --pass-pipeline='builtin.module(ensure-debug-info-scope-on-llvm-func,gpu-module-to-binary{format=isa})' --remarks-filter-passed=llvm-inline 2>&1 | FileCheck %s --check-prefix=CHECK-LOC
// RUN: mlir-opt %s --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' --remarks-filter-passed=llvm-loop-unroll 2>&1 | FileCheck %s --check-prefix=CHECK-NONE
// RUN: mlir-opt %s --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' --remarks-filter-missed=llvm-inline 2>&1 | FileCheck %s --check-prefix=CHECK-NONE

// Remarks emitted by LLVM while serializing the GPU module are reported through
// the MLIR remark engine, with the LLVM pass name, prefixed with `llvm-`, as the category.

// CHECK-NONE-NOT: remark:
// CHECK-NONE: gpu.binary @kernel_module
// CHECK-NONE-NOT: remark:

module attributes {gpu.container_module} {
  gpu.module @kernel_module [#nvvm.target<chip = "sm_70">] {
    llvm.func @helper(%a: f32, %b: f32) -> f32 {
      %0 = llvm.fadd %a, %b : f32
      llvm.return %0 : f32
    }
    // Without debug info, the remark is attached to the function.
    // CHECK: remarks.mlir:[[#@LINE+1]]:{{[0-9]+}}: remark: [Passed] Inlined | Category:llvm-inline | Function=kernel | Callee=helper, Caller=kernel, Cost={{.*}}, Remark="'helper' inlined into 'kernel'
    llvm.func @kernel(%arg0: f32, %arg1: !llvm.ptr) attributes {gpu.kernel} {
      // With debug info, the remark is attached to the call.
      // CHECK-LOC: remarks.mlir:[[#@LINE+1]]:{{[0-9]+}}: remark: [Passed] Inlined | Category:llvm-inline | Function=kernel
      %0 = llvm.call @helper(%arg0, %arg0) : (f32, f32) -> f32
      llvm.store %0, %arg1 : f32, !llvm.ptr
      llvm.return
    }
  }
}
