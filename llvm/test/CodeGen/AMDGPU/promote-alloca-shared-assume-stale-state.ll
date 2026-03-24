; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 %s | FileCheck %s

declare void @llvm.assume(i1)

; Regression test for stale state after shared-assume rewriting.
; Earlier promotions can erase users shared with later allocas. Re-analysis
; must use current IR and skip stale promotion state.

define amdgpu_kernel void @skip_dead_alloca_after_shared_assume(ptr addrspace(1) %out) #0 {
; CHECK-LABEL: @skip_dead_alloca_after_shared_assume(
; CHECK: alloca [4 x i32], align 4, addrspace(5)
; CHECK: freeze <4 x i32> poison
; CHECK-NOT: freeze <4 x i32> poison
; CHECK: call void @llvm.assume(i1 true)
; CHECK: store i32 7, ptr addrspace(1) %out, align 4
; CHECK: ret void
entry:
  %a = alloca [4 x i32], align 4, addrspace(5)
  %b = alloca [4 x i32], align 4, addrspace(5)

  ; This compare is shared by %a and %b.
  %cmp = icmp ne ptr addrspace(5) %a, %b
  call void @llvm.assume(i1 %cmp)

  ; Make %a high-score so it is promoted first.
  %a0 = getelementptr inbounds i32, ptr addrspace(5) %a, i32 0
  store i32 7, ptr addrspace(5) %a0, align 4
  %av = load i32, ptr addrspace(5) %a0, align 4
  store i32 %av, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { "amdgpu-promote-alloca-to-vector-max-regs"="1024" }
