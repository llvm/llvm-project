// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @llvm_nvvm_fence_sc_cluster
llvm.func @llvm_nvvm_fence_sc_cluster() {
  // CHECK: nvvm.fence.sc.cluster
  nvvm.fence.sc.cluster
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_sync_restrict
llvm.func @nvvm_fence_sync_restrict() {
  // CHECK: call void @llvm.nvvm.fence.acquire.sync_restrict.space.cluster.scope.cluster()
  nvvm.fence.sync_restrict {order = #nvvm.mem_order<acquire>}
  // CHECK: call void @llvm.nvvm.fence.release.sync_restrict.space.cta.scope.cluster()
  nvvm.fence.sync_restrict {order = #nvvm.mem_order<release>}
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
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<alias>}

  // CHECK: call void @llvm.nvvm.fence.proxy.async()
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<async>}

  // CHECK: call void @llvm.nvvm.fence.proxy.async.global()
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.global>}

  // CHECK: call void @llvm.nvvm.fence.proxy.async.shared_cta()
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}

  // CHECK: call void @llvm.nvvm.fence.proxy.async.shared_cluster()
  nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cluster>}
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_sync_restrict
llvm.func @nvvm_fence_proxy_sync_restrict() {
  // CHECK: call void @llvm.nvvm.fence.proxy.async_generic.acquire.sync_restrict.space.cluster.scope.cluster()
  nvvm.fence.proxy.sync_restrict {order = #nvvm.mem_order<acquire>}
  // CHECK: call void @llvm.nvvm.fence.proxy.async_generic.release.sync_restrict.space.cta.scope.cluster()
  nvvm.fence.proxy.sync_restrict {order = #nvvm.mem_order<release>}
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_release
llvm.func @nvvm_fence_proxy_tensormap_generic_release() {
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cta()
  nvvm.fence.proxy.release #nvvm.mem_scope<cta>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cluster()
  nvvm.fence.proxy.release #nvvm.mem_scope<cluster>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.gpu()
  nvvm.fence.proxy.release #nvvm.mem_scope<gpu>

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.release.sys()
  nvvm.fence.proxy.release #nvvm.mem_scope<sys>
  llvm.return
}

// CHECK-LABEL: @nvvm_fence_proxy_tensormap_generic_acquire
llvm.func @nvvm_fence_proxy_tensormap_generic_acquire(%addr : !llvm.ptr) {
  %c128 = llvm.mlir.constant(128) : i32
  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cta> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<cluster> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<gpu> %addr, %c128

  // CHECK: call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr {{%[0-9]+}}, i32 128)
  nvvm.fence.proxy.acquire #nvvm.mem_scope<sys> %addr, %c128
  llvm.return
}
