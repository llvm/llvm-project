; RUN: not llvm-as -disable-output < %s 2>&1 | FileCheck %s

define void @test_fence_proxy_tensormap_generic_acquire(ptr addrspace(0) %addr) {
  ; CHECK: immarg value 130 for arg 1 out of range [128,129)
  call void @llvm.nvvm.fence.proxy.tensormap_generic.acquire.cta(ptr addrspace(0) %addr, i32 130);

  ret void
}
