; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.brev(i32) nounwind readnone

; FUNC-LABEL: {{^}}s_brev_i32:
; SI: s_load_dword [[VAL:s[0-9]+]],
; SI: s_brev_b32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: buffer_store_dword [[VRESULT]],
; SI: s_endpgm
define void @s_brev_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctlz = call i32 @llvm.AMDGPU.brev(i32 %val) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_brev_i32:
; SI: buffer_load_dword [[VAL:v[0-9]+]],
; SI: v_bfrev_b32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @v_brev_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr, align 4
  %ctlz = call i32 @llvm.AMDGPU.brev(i32 %val) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}
