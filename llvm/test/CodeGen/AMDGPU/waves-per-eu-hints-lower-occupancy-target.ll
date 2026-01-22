; RUN: opt -S -mtriple=amdgcn-unknown-unknown -mcpu=tahiti -passes=amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck %s

; All kernels have the same value for amdgpu-flat-work-group-size, except the
; second and third kernels explicitly set it. The first and second kernel also
; have the same final waves/EU range, except the second kernel explicitly sets
; it with amdgpu-waves-per-eu. As a result the first and second kernels are
; treated identically. 
;
; The third kernel hints the compiler that a maximum occupancy of 1 is desired
; with amdgpu-waves-per-eu, so the alloca promotion pass is free to use more LDS
; space than when limiting itself to support the maximum default occupancy of
; 10. This does not break the ABI requirement to support the full possible range
; of workgroup sizes as specified by amdgpu-flat-work-group-size.

; CHECK-NOT: @no_attributes.stack
; CHECK-NOT: @explicit_default_workgroup_size_and_waves.stack

; CHECK-LABEL: @no_attributes(
; CHECK: alloca [5 x i32]
; CHECK: store i32 4, ptr addrspace(5) %arrayidx1, align 4
define amdgpu_kernel void @no_attributes(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %in_data0 = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data0
  store i32 4, ptr addrspace(5) %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %in_data1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data1
  store i32 5, ptr addrspace(5) %arrayidx3, align 4
  %out_data0 = load i32, ptr addrspace(5) %stack, align 4
  store i32 %out_data0, ptr addrspace(1) %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %out_data1 = load i32, ptr addrspace(5) %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 1
  store i32 %out_data1, ptr addrspace(1) %arrayidx13
  ret void
}

; CHECK-LABEL: @explicit_default_workgroup_size_and_waves(
; CHECK: alloca [5 x i32]
; CHECK: store i32 4, ptr addrspace(5) %arrayidx1, align 4
define amdgpu_kernel void @explicit_default_workgroup_size_and_waves(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %in_data0 = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data0
  store i32 4, ptr addrspace(5) %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %in_data1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data1
  store i32 5, ptr addrspace(5) %arrayidx3, align 4
  %out_data0 = load i32, ptr addrspace(5) %stack, align 4
  store i32 %out_data0, ptr addrspace(1) %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %out_data1 = load i32, ptr addrspace(5) %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 1
  store i32 %out_data1, ptr addrspace(1) %arrayidx13
  ret void
}

; CHECK-LABEL: @explicit_low_occupancy_requested(
; CHECK-NOT: alloca [5 x i32]
define amdgpu_kernel void @explicit_low_occupancy_requested(ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %in) #1 {
entry:
  %stack = alloca [5 x i32], align 4, addrspace(5)
  %in_data0 = load i32, ptr addrspace(1) %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data0
  store i32 4, ptr addrspace(5) %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %in_data1 = load i32, ptr addrspace(1) %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 %in_data1
  store i32 5, ptr addrspace(5) %arrayidx3, align 4
  %out_data0 = load i32, ptr addrspace(5) %stack, align 4
  store i32 %out_data0, ptr addrspace(1) %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], ptr addrspace(5) %stack, i32 0, i32 1
  %out_data1 = load i32, ptr addrspace(5) %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, ptr addrspace(1) %out, i32 1
  store i32 %out_data1, ptr addrspace(1) %arrayidx13
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="4,10" "amdgpu-flat-work-group-size"="1,1024" }
attributes #1 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,1024" }
