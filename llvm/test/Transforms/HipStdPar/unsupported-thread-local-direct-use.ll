; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %s 2>&1 | FileCheck %s

@tls = hidden thread_local addrspace(1) global i32 0, align 4

; CHECK: error: {{.*}} in function direct_use void (): Accelerator does not support the thread_local variable tls
define amdgpu_kernel void @direct_use() {
entry:
  %0 = call align 4 ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) @tls)
  %1 = load i32, ptr addrspace(1) %0, align 4
  ret void
}

declare nonnull ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) nonnull)
