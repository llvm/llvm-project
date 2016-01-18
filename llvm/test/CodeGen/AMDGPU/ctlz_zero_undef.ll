; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1) nounwind readnone
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1) nounwind readnone

; FUNC-LABEL: {{^}}s_ctlz_zero_undef_i32:
; SI: s_load_dword [[VAL:s[0-9]+]],
; SI: s_flbit_i32_b32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: buffer_store_dword [[VRESULT]],
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define void @s_ctlz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32:
; SI: buffer_load_dword [[VAL:v[0-9]+]],
; SI: v_ffbh_u32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define void @v_ctlz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr, align 4
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_v2i32:
; SI: buffer_load_dwordx2
; SI: v_ffbh_u32_e32
; SI: v_ffbh_u32_e32
; SI: buffer_store_dwordx2
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define void @v_ctlz_zero_undef_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %valptr, align 8
  %ctlz = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %val, i1 true) nounwind readnone
  store <2 x i32> %ctlz, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_v4i32:
; SI: buffer_load_dwordx4
; SI: v_ffbh_u32_e32
; SI: v_ffbh_u32_e32
; SI: v_ffbh_u32_e32
; SI: v_ffbh_u32_e32
; SI: buffer_store_dwordx4
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define void @v_ctlz_zero_undef_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %valptr, align 16
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %val, i1 true) nounwind readnone
  store <4 x i32> %ctlz, <4 x i32> addrspace(1)* %out, align 16
  ret void
}
