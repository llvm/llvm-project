; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-promote-alloca -disable-promote-alloca-to-vector < %s | FileCheck %s

; This kernel starts with the amdgpu-no-workitem-id-* attributes, but
; need to be removed when these intrinsic uses are introduced.

; CHECK-LABEL: define amdgpu_kernel void @promote_to_lds(i32 addrspace(1)* %out, i32 %in) #0 {
; CHECK: call noalias nonnull dereferenceable(64) i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr()
; CHECK: call i32 @llvm.amdgcn.workitem.id.x(), !range !2
; CHECK: call i32 @llvm.amdgcn.workitem.id.y(), !range !2
; CHECK: call i32 @llvm.amdgcn.workitem.id.z(), !range !2
define amdgpu_kernel void @promote_to_lds(i32 addrspace(1)* %out, i32 %in) #0 {
entry:
  %tmp = alloca [2 x i32], addrspace(5)
  %tmp1 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %tmp, i32 0, i32 0
  %tmp2 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %tmp, i32 0, i32 1
  store i32 0, i32 addrspace(5)* %tmp1
  store i32 1, i32 addrspace(5)* %tmp2
  %tmp3 = getelementptr inbounds [2 x i32], [2 x i32] addrspace(5)* %tmp, i32 0, i32 %in
  %tmp4 = load i32, i32 addrspace(5)* %tmp3
  %tmp5 = load volatile i32, i32 addrspace(1)* undef
  %tmp6 = add i32 %tmp4, %tmp5
  store i32 %tmp6, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1,5" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-dispatch-ptr" }

; CHECK: attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1,5" }
