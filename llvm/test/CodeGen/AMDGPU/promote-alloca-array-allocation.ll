; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca < %s | FileCheck %s

; Make sure this allocates the correct size if the alloca has a non-0
; number of elements.

; CHECK-LABEL: @array_alloca(
; CHECK: %stack = alloca i32, i32 5, align 4, addrspace(5)
define amdgpu_kernel void @array_alloca(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) #0 {
entry:
  %stack = alloca i32, i32 5, align 4, addrspace(5)
  %ld0 = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 %ld0
  store i32 4, ptr addrspace(5) %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %ld1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 %ld1
  store i32 5, ptr addrspace(5) %arrayidx3, align 4
  %ld2 = load i32, ptr addrspace(5) %stack, align 4
  store i32 %ld2, ptr addrspace(1) %out, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 1
  %ld3 = load i32, ptr addrspace(5) %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 1
  store i32 %ld3, ptr addrspace(1) %arrayidx13
  ret void
}

; CHECK-LABEL: @array_alloca_dynamic(
; CHECK: %stack = alloca i32, i32 %size, align 4, addrspace(5)
define amdgpu_kernel void @array_alloca_dynamic(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in, i32 %size) #0 {
entry:
  %stack = alloca i32, i32 %size, align 4, addrspace(5)
  %ld0 = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 %ld0
  store i32 4, ptr addrspace(5) %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %ld1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 %ld1
  store i32 5, ptr addrspace(5) %arrayidx3, align 4
  %ld2 = load i32, ptr addrspace(5) %stack, align 4
  store i32 %ld2, ptr addrspace(1) %out, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr addrspace(5) %stack, i32 1
  %ld3 = load i32, ptr addrspace(5) %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 1
  store i32 %ld3, ptr addrspace(1) %arrayidx13
  ret void
}

attributes #0 = { nounwind }
