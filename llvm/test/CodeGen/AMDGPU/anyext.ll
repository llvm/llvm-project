; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() nounwind readnone

; GCN-LABEL: {{^}}anyext_i1_i32:
; GCN: v_cndmask_b32_e64
define amdgpu_kernel void @anyext_i1_i32(i32 addrspace(1)* %out, i32 %cond) {
entry:
  %tmp = icmp eq i32 %cond, 0
  %tmp1 = zext i1 %tmp to i8
  %tmp2 = xor i8 %tmp1, -1
  %tmp3 = and i8 %tmp2, 1
  %tmp4 = zext i8 %tmp3 to i32
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_anyext_i16_i32:
; VI: v_add_u16_e32 [[ADD:v[0-9]+]],
; VI: v_xor_b32_e32 [[XOR:v[0-9]+]], -1, [[ADD]]
; VI: v_and_b32_e32 [[AND:v[0-9]+]], 1, [[XOR]]
; VI: buffer_store_dword [[AND]]
define amdgpu_kernel void @s_anyext_i16_i32(i32 addrspace(1)* %out, i16 addrspace(1)* %a, i16 addrspace(1)* %b) {
entry:
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.y = call i32 @llvm.amdgcn.workitem.id.y()
  %a.ptr = getelementptr i16, i16 addrspace(1)* %a, i32 %tid.x
  %b.ptr = getelementptr i16, i16 addrspace(1)* %b, i32 %tid.y
  %a.l = load i16, i16 addrspace(1)* %a.ptr
  %b.l = load i16, i16 addrspace(1)* %b.ptr
  %tmp = add i16 %a.l, %b.l
  %tmp1 = trunc i16 %tmp to i8
  %tmp2 = xor i8 %tmp1, -1
  %tmp3 = and i8 %tmp2, 1
  %tmp4 = zext i8 %tmp3 to i32
  store i32 %tmp4, i32 addrspace(1)* %out
  ret void
}
