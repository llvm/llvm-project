; RUN: not opt -S -mtriple=amdgcn-amd-amdhsa -passes=hipstdpar-select-accelerator-code \
; RUN:   %s 2>&1 | FileCheck %s

@tls = hidden thread_local addrspace(1) global i32 0, align 4

; CHECK: error: {{.*}} in function indirect_use void (): Accelerator does not support the thread_local variable tls
define amdgpu_kernel void @indirect_use() {
entry:
  %0 = call align 4 ptr @llvm.threadlocal.address.p0(ptr addrspacecast (ptr addrspace(1) @tls to ptr))
  %1 = load i32, ptr %0, align 4
  ret void
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)
