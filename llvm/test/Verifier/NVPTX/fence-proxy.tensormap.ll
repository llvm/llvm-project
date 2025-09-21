; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @test_fence_proxy_tensormap_generic_acquire(ptr addrspace(0) %addr) {
  ; CHECK: immarg value 127 out of range [128, 129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr addrspace(0) %addr, i32 127);

  ; CHECK: immarg value 129 out of range [128, 129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cluster(ptr addrspace(0) %addr, i32 129);

  ; CHECK: immarg value 127 out of range [128, 129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.gpu(ptr addrspace(0) %addr, i32 127);

  ; CHECK: immarg value 129 out of range [128, 129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys(ptr addrspace(0) %addr, i32 129);

  ret void
}
