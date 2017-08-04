; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-promote-alloca < %s | FileCheck %s

; Make sure this allocates the correct size if the alloca has a non-0
; number of elements.

; CHECK-LABEL: @array_alloca(
; CHECK: %stack = alloca i32, i32 5, align 4
define amdgpu_kernel void @array_alloca(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca i32, i32 5, align 4
  %ld0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %stack, i32 %ld0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %ld1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %stack, i32 %ld1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %stack, i32 0
  %ld2 = load i32, i32* %arrayidx10, align 4
  store i32 %ld2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds i32, i32* %stack, i32 1
  %ld3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %ld3, i32 addrspace(1)* %arrayidx13
  ret void
}

; CHECK-LABEL: @array_alloca_dynamic(
; CHECK: %stack = alloca i32, i32 %size, align 4
define amdgpu_kernel void @array_alloca_dynamic(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in, i32 %size) #0 {
entry:
  %stack = alloca i32, i32 %size, align 4
  %ld0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %stack, i32 %ld0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %ld1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %stack, i32 %ld1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %stack, i32 0
  %ld2 = load i32, i32* %arrayidx10, align 4
  store i32 %ld2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds i32, i32* %stack, i32 1
  %ld3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %ld3, i32 addrspace(1)* %arrayidx13
  ret void
}

attributes #0 = { nounwind }
