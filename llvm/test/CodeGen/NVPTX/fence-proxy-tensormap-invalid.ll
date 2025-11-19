; RUN: not llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx83 -o /dev/null 2>&1 | FileCheck %s

define void @test_fence_proxy_tensormap_generic_acquire(ptr addrspace(0) %addr) {
  ; CHECK: immarg value 130 out of range [128, 129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr addrspace(0) %addr, i32 130);

  ret void
}
