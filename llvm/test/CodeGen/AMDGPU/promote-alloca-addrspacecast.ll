; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck %s

; The types of the users of the addrspacecast should not be changed.

; CHECK-LABEL: @invalid_bitcast_addrspace(
; CHECK: [[GEP:%[0-9]+]] = getelementptr inbounds [256 x [1 x i32]], ptr addrspace(3) @invalid_bitcast_addrspace.data, i32 0, i32 %{{[0-9]+}}
; CHECK: [[ASC:%[a-z0-9]+]] = addrspacecast ptr addrspace(3) [[GEP]] to ptr
; CHECK: [[LOAD:%[a-z0-9]+]] = load <2 x i16>, ptr [[ASC]]
; CHECK: bitcast <2 x i16> [[LOAD]] to <2 x half>
define amdgpu_kernel void @invalid_bitcast_addrspace() #0 {
entry:
  %data = alloca [1 x i32], addrspace(5)
  %tmp1 = addrspacecast ptr addrspace(5) %data to ptr
  %tmp3 = load <2 x i16>, ptr %tmp1, align 2
  %tmp4 = bitcast <2 x i16> %tmp3 to <2 x half>
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" }
