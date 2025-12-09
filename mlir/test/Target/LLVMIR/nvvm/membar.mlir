// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-lABEL: @memorybarrier()
llvm.func @memorybarrier() {
  // CHECK: call void @llvm.nvvm.membar.cta()
  nvvm.memory.barrier #nvvm.mem_scope<cta>
  // CHECK: call void @llvm.nvvm.fence.sc.cluster()
  nvvm.memory.barrier #nvvm.mem_scope<cluster>
  // CHECK: call void @llvm.nvvm.membar.gl()
  nvvm.memory.barrier #nvvm.mem_scope<gpu>
  // CHECK: call void @llvm.nvvm.membar.sys()
  nvvm.memory.barrier #nvvm.mem_scope<sys>
  llvm.return
}
