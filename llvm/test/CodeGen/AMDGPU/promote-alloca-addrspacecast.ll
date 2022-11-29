; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -passes=amdgpu-promote-alloca < %s | FileCheck %s

; The types of the users of the addrspacecast should not be changed.

; CHECK-LABEL: @invalid_bitcast_addrspace(
; CHECK: getelementptr inbounds [256 x [1 x i32]], [256 x [1 x i32]] addrspace(3)* @invalid_bitcast_addrspace.data, i32 0, i32 %14
; CHECK: bitcast [1 x i32] addrspace(3)* %{{[0-9]+}} to half addrspace(3)*
; CHECK: addrspacecast half addrspace(3)* %tmp to half*
; CHECK: bitcast half* %tmp1 to <2 x i16>*
define amdgpu_kernel void @invalid_bitcast_addrspace() #0 {
entry:
  %data = alloca [1 x i32], addrspace(5)
  %tmp = bitcast [1 x i32] addrspace(5)* %data to half addrspace(5)*
  %tmp1 = addrspacecast half addrspace(5)* %tmp to half*
  %tmp2 = bitcast half* %tmp1 to <2 x i16>*
  %tmp3 = load <2 x i16>, <2 x i16>* %tmp2, align 2
  %tmp4 = bitcast <2 x i16> %tmp3 to <2 x half>
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" }
