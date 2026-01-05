// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-LABEL: @llvm_nvvm_tcgen05_fence
llvm.func @llvm_nvvm_tcgen05_fence() {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.fence.before.thread.sync()
  nvvm.tcgen05.fence #nvvm.tcgen05_fence<before>

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.fence.after.thread.sync()
  nvvm.tcgen05.fence #nvvm.tcgen05_fence<after>

  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_wait
llvm.func @llvm_nvvm_tcgen05_wait() {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.wait.ld()
  nvvm.tcgen05.wait #nvvm.tcgen05_wait<load>

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.wait.st()
  nvvm.tcgen05.wait #nvvm.tcgen05_wait<store>

  llvm.return
}
