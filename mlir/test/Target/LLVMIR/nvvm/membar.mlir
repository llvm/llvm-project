// RUN: mlir-translate -mlir-to-llvmir %s  -split-input-file --verify-diagnostics | FileCheck %s

// CHECK-lABEL: @memorybarrier()
llvm.func @memorybarrier() {
  // CHECK: call void @llvm.nvvm.membar.cta()
  nvvm.memory.barrier cta
  // CHECK: call void @llvm.nvvm.fence.sc.cluster()
  nvvm.memory.barrier cluster
  // CHECK: call void @llvm.nvvm.membar.gl()
  nvvm.memory.barrier gpu
  // CHECK: call void @llvm.nvvm.membar.sys()
  nvvm.memory.barrier sys
  llvm.return
}
