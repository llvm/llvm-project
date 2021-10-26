; RUN: opt -S -mtriple=amdgcn-unknown-unknown -mcpu=tahiti -amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck %s

; Both of these kernels have the same value for
; amdgpu-flat-work-group-size, except one explicitly sets it. This is
; a program visible property which should always take precedence over
; the amdgpu-waves-per-eu optimization hint.
;
; The range is incompatible with the amdgpu-waves-per-eu value, so the
; flat work group size should take precedence implying a requirement
; to support 1024 size workgroups (which exceeds the available LDS
; amount).

; CHECK-NOT: @no_flat_workgroup_size.stack
; CHECK-NOT: @explicit_default_workgroup_size.stack

; CHECK-LABEL: @no_flat_workgroup_size(
; CHECK: alloca [5 x i32]
; CHECK: store i32 4, i32 addrspace(5)* %arrayidx1, align 4
define amdgpu_kernel void @no_flat_workgroup_size(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %0
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 0
  %2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 1
  %3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

; CHECK-LABEL: @explicit_default_workgroup_size(
; CHECK: alloca [5 x i32]
; CHECK: store i32 4, i32 addrspace(5)* %arrayidx1, align 4
define amdgpu_kernel void @explicit_default_workgroup_size(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #1 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %0
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 %1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 0
  %2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32] addrspace(5)* %stack, i32 0, i32 1
  %3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" }
attributes #1 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,1024" }
