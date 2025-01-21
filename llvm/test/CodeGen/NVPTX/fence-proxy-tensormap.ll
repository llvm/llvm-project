; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-12.3 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 | %ptxas-verify -arch=sm_90 %}

; CHECK-LABEL: test_fence_proxy_tensormap_generic_release
define void @test_fence_proxy_tensormap_generic_release() {
  ; CHECK: fence.proxy.tensormap::generic.release.cta;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cta();

  ; CHECK: fence.proxy.tensormap::generic.release.cluster;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.release.cluster();

  ; CHECK: fence.proxy.tensormap::generic.release.gpu;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.release.gpu();

  ; CHECK: fence.proxy.tensormap::generic.release.sys;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.release.sys();

  ret void
}

; CHECK-LABEL: test_fence_proxy_tensormap_generic_acquire
define void @test_fence_proxy_tensormap_generic_acquire(ptr addrspace(0) %addr) {
  ; CHECK: fence.proxy.tensormap::generic.acquire.cta [%rd{{[0-9]+}}], 128;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr addrspace(0) %addr, i32 128);

  ; CHECK: fence.proxy.tensormap::generic.acquire.cluster [%rd{{[0-9]+}}], 128;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr addrspace(0) %addr, i32 128);

  ; CHECK: fence.proxy.tensormap::generic.acquire.gpu [%rd{{[0-9]+}}], 128;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr addrspace(0) %addr, i32 128);

  ; CHECK: fence.proxy.tensormap::generic.acquire.sys [%rd{{[0-9]+}}], 128;
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr addrspace(0) %addr, i32 128);

  ret void
}
