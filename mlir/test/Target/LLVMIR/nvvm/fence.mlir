// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s
// RUN: mlir-opt %s | FileCheck %s --check-prefix=CHECK-ASM

// CHECK-LABEL: @llvm_nvvm_fence_sc_cluster
llvm.func @llvm_nvvm_fence_sc_cluster() {
  // CHECK: nvvm.fence.sc.cluster
  nvvm.fence.sc.cluster
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_sync_restrict
llvm.func @nvvm_fence_sync_restrict() {
  // CHECK-ASM: nvvm.fence.sync_restrict <acquire>
  // CHECK: call void @llvm.nvvm.fence.acquire.sync_restrict.space.cluster.scope.cluster()
  nvvm.fence.sync_restrict <acquire>
  // CHECK-ASM: nvvm.fence.sync_restrict <release>
  // CHECK: call void @llvm.nvvm.fence.release.sync_restrict.space.cta.scope.cluster()
  nvvm.fence.sync_restrict <release>
  llvm.return
}

// CHECK-LABEL: @fence_mbarrier_init
llvm.func @fence_mbarrier_init() {
  // CHECK: call void @llvm.nvvm.fence.mbarrier_init.release.cluster()
  nvvm.fence.mbarrier.init
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy
llvm.func @nvvm_fence_proxy() {
  // CHECK: call void @llvm.nvvm.fence.proxy.alias()
  nvvm.fence.proxy <alias>

  // CHECK: call void @llvm.nvvm.fence.proxy.async()
  nvvm.fence.proxy <async>

  // CHECK: call void @llvm.nvvm.fence.proxy.async.global()
  nvvm.fence.proxy <async.global>

  // CHECK: call void @llvm.nvvm.fence.proxy.async.shared_cta()
  // CHECK-ASM: nvvm.fence.proxy <async.shared>, space = <cta>
  nvvm.fence.proxy <async.shared>, space = <cta>

  // CHECK: call void @llvm.nvvm.fence.proxy.async.shared_cluster()
  // CHECK-ASM: nvvm.fence.proxy <async.shared>, space = <cluster>
  nvvm.fence.proxy <async.shared>, space = <cluster>
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_sync_restrict
llvm.func @nvvm_fence_proxy_sync_restrict() {
  // CHECK: call void @llvm.nvvm.fence.proxy.async_generic.acquire.sync_restrict.space.cluster.scope.cluster()
  nvvm.fence.proxy.sync_restrict <acquire>
  // CHECK: call void @llvm.nvvm.fence.proxy.async_generic.release.sync_restrict.space.cta.scope.cluster()
  nvvm.fence.proxy.sync_restrict <release>
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_release
llvm.func @nvvm_fence_proxy_tensormap_generic_release() {
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cta()
  nvvm.fence.proxy.release cta

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cluster()
  nvvm.fence.proxy.release cluster

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.gpu()
  nvvm.fence.proxy.release gpu

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.sys()
  nvvm.fence.proxy.release sys
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_acquire
llvm.func @nvvm_fence_proxy_tensormap_generic_acquire(%addr : !llvm.ptr) {
  %c128 = llvm.mlir.constant(128) : i32
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire cta %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire cluster %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire gpu %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire sys %addr, %c128
  llvm.return
}
