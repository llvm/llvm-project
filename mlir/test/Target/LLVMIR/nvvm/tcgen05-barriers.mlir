// RUN: mlir-opt -split-input-file -verify-diagnostics %s
// RUN: mlir-translate -mlir-to-llvmir -split-input-file -verify-diagnostics %s | FileCheck %s --check-prefix=CHECK-LLVM

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

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_generic
llvm.func @llvm_nvvm_tcgen05_commit_generic(%barrier : !llvm.ptr, %cta_mask : i16) {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.cg1(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier : !llvm.ptr

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.cg2(ptr %{{.*}})
  nvvm.tcgen05.commit %barrier {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.mc.cg1(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask : !llvm.ptr, i16

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.mc.cg2(ptr %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr, i16
  llvm.return
}

// CHECK-LABEL: @llvm_nvvm_tcgen05_commit_shared
llvm.func @llvm_nvvm_tcgen05_commit_shared(%barrier : !llvm.ptr<3>, %cta_mask : i16) {
  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.shared.cg1(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier : !llvm.ptr<3>

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.shared.cg2(ptr addrspace(3) %{{.*}})
  nvvm.tcgen05.commit %barrier {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr<3>

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.mc.shared.cg1(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask : !llvm.ptr<3>, i16

  // CHECK-LLVM: call void @llvm.nvvm.tcgen05.commit.mc.shared.cg2(ptr addrspace(3) %{{.*}}, i16 %{{.*}})
  nvvm.tcgen05.commit %barrier, multicast_mask = %cta_mask {group = #nvvm.tcgen05_group<cta_2>} : !llvm.ptr<3>, i16
  llvm.return
}
