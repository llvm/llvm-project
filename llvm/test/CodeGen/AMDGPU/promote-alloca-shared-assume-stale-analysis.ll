; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca -amdgpu-promote-alloca-to-vector-limit=512 %s | FileCheck %s

declare void @llvm.assume(i1)

; Check that there is no use-after-free when promoting one alloca
; touches instructions that also relate to another alloca.

; Two allocas sharing an icmp+assume chain.

define amdgpu_kernel void @shared_assume_stale_analysis_two_allocas(ptr addrspace(1) %out) {
; CHECK-LABEL: @shared_assume_stale_analysis_two_allocas(
; CHECK-NOT: alloca
; CHECK: call void @llvm.assume(i1 true)
; CHECK: ret void
entry:
  %a = alloca [4 x i32], align 4, addrspace(5)
  %b = alloca [4 x i32], align 4, addrspace(5)

  %a0 = getelementptr inbounds i32, ptr addrspace(5) %a, i32 0
  %b0 = getelementptr inbounds i32, ptr addrspace(5) %b, i32 0
  %cmp = icmp ne ptr addrspace(5) %a0, %b0
  call void @llvm.assume(i1 %cmp)

  store i32 1, ptr addrspace(5) %a0, align 4
  %av = load i32, ptr addrspace(5) %a0, align 4
  store i32 2, ptr addrspace(5) %b0, align 4
  %bv = load i32, ptr addrspace(5) %b0, align 4
  %sum = add i32 %av, %bv
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

; Three allocas where the middle alloca participates in two icmp+assume chains.
; Promoting it first erases both chains.

define amdgpu_kernel void @shared_assume_stale_analysis_three_allocas(ptr addrspace(1) %out) {
; CHECK-LABEL: @shared_assume_stale_analysis_three_allocas(
; CHECK-NOT: alloca
; CHECK: ret void
entry:
  %a = alloca [4 x i32], align 4, addrspace(5)
  %b = alloca [4 x i32], align 4, addrspace(5)
  %c = alloca [4 x i32], align 4, addrspace(5)

  %a0 = getelementptr inbounds i32, ptr addrspace(5) %a, i32 0
  %b0 = getelementptr inbounds i32, ptr addrspace(5) %b, i32 0
  %c0 = getelementptr inbounds i32, ptr addrspace(5) %c, i32 0
  %cmp.ab = icmp ne ptr addrspace(5) %a0, %b0
  call void @llvm.assume(i1 %cmp.ab)
  %cmp.bc = icmp ne ptr addrspace(5) %b0, %c0
  call void @llvm.assume(i1 %cmp.bc)

  store i32 1, ptr addrspace(5) %a0, align 4
  %av = load i32, ptr addrspace(5) %a0, align 4
  store i32 2, ptr addrspace(5) %b0, align 4
  %bv = load i32, ptr addrspace(5) %b0, align 4
  store i32 3, ptr addrspace(5) %c0, align 4
  %cv = load i32, ptr addrspace(5) %c0, align 4
  %sum0 = add i32 %av, %bv
  %sum1 = add i32 %sum0, %cv
  store i32 %sum1, ptr addrspace(1) %out, align 4
  ret void
}
