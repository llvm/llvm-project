; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -amdgpu-promote-alloca < %s | FileCheck %s

; The types of the users of the addrspacecast should not be changed.

; CHECK-LABEL: @invalid_bitcast_addrspace(
; CHECK: getelementptr inbounds [256 x [1 x i32]], [256 x [1 x i32]] addrspace(3)* @invalid_bitcast_addrspace.data, i32 0, i32 %14
; CHECK: bitcast [1 x i32] addrspace(3)* %{{[0-9]+}} to half addrspace(3)*
; CHECK: addrspacecast half addrspace(3)* %tmp to half addrspace(4)*
; CHECK: bitcast half addrspace(4)* %tmp1 to <2 x i16> addrspace(4)*
define amdgpu_kernel void @invalid_bitcast_addrspace() #0 {
entry:
  %data = alloca [1 x i32], align 4
  %tmp = bitcast [1 x i32]* %data to half*
  %tmp1 = addrspacecast half* %tmp to half addrspace(4)*
  %tmp2 = bitcast half addrspace(4)* %tmp1 to <2 x i16> addrspace(4)*
  %tmp3 = load <2 x i16>, <2 x i16> addrspace(4)* %tmp2, align 2
  %tmp4 = bitcast <2 x i16> %tmp3 to <2 x half>
  ret void
}

attributes #0 = { nounwind }
