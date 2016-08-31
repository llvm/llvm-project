; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=cypress < %s | FileCheck -check-prefix=EG %s

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

define void @trunc_i64_to_i32_store(i32 addrspace(1)* %out, i64 %in) {
; SI-LABEL: {{^}}trunc_i64_to_i32_store:
; SI: s_load_dword [[SLOAD:s[0-9]+]], s[0:1], 0xb
; SI: v_mov_b32_e32 [[VLOAD:v[0-9]+]], [[SLOAD]]
; SI: buffer_store_dword [[VLOAD]]

; EG-LABEL: {{^}}trunc_i64_to_i32_store:
; EG: MEM_RAT_CACHELESS STORE_RAW T0.X, T1.X, 1
; EG: LSHR
; EG-NEXT: 2(

  %result = trunc i64 %in to i32 store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}trunc_load_shl_i64:
; SI-DAG: s_load_dwordx2
; SI-DAG: s_load_dword [[SREG:s[0-9]+]],
; SI: s_lshl_b32 [[SHL:s[0-9]+]], [[SREG]], 2
; SI: v_mov_b32_e32 [[VSHL:v[0-9]+]], [[SHL]]
; SI: buffer_store_dword [[VSHL]],
define void @trunc_load_shl_i64(i32 addrspace(1)* %out, i64 %a) {
  %b = shl i64 %a, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}trunc_shl_i64:
; SI: s_load_dwordx2 s{{\[}}[[LO_SREG:[0-9]+]]:{{[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI: s_lshl_b64 s{{\[}}[[LO_SHL:[0-9]+]]:{{[0-9]+\]}}, s{{\[}}[[LO_SREG]]:{{[0-9]+\]}}, 2
; SI: s_add_u32 s[[LO_SREG2:[0-9]+]], s[[LO_SHL]],
; SI: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], s[[LO_SREG2]]
; SI: s_addc_u32
; SI: buffer_store_dword v[[LO_VREG]],
; SI: v_mov_b32_e32
; SI: v_mov_b32_e32
define void @trunc_shl_i64(i64 addrspace(1)* %out2, i32 addrspace(1)* %out, i64 %a) {
  %aa = add i64 %a, 234 ; Prevent shrinking store.
  %b = shl i64 %aa, 2
  %result = trunc i64 %b to i32
  store i32 %result, i32 addrspace(1)* %out, align 4
  store i64 %b, i64 addrspace(1)* %out2, align 8 ; Prevent reducing ops to 32-bits
  ret void
}

; SI-LABEL: {{^}}trunc_i32_to_i1:
; SI: v_and_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; SI: v_cmp_eq_i32
define void @trunc_i32_to_i1(i32 addrspace(1)* %out, i32 addrspace(1)* %ptr) {
  %a = load i32, i32 addrspace(1)* %ptr, align 4
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}sgpr_trunc_i32_to_i1:
; SI: s_and_b32 s{{[0-9]+}}, 1, s{{[0-9]+}}
; SI: v_cmp_eq_i32
define void @sgpr_trunc_i32_to_i1(i32 addrspace(1)* %out, i32 %a) {
  %trunc = trunc i32 %a to i1
  %result = select i1 %trunc, i32 1, i32 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}s_trunc_i64_to_i1:
; SI: s_load_dwordx2 s{{\[}}[[SLO:[0-9]+]]:{{[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI: s_and_b32 [[MASKED:s[0-9]+]], 1, s[[SLO]]
; SI: v_cmp_eq_i32_e64 s{{\[}}[[VLO:[0-9]+]]:[[VHI:[0-9]+]]], 1, [[MASKED]]
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, -12, 63, s{{\[}}[[VLO]]:[[VHI]]]
define void @s_trunc_i64_to_i1(i32 addrspace(1)* %out, i64 %x) {
  %trunc = trunc i64 %x to i1
  %sel = select i1 %trunc, i32 63, i32 -12
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}v_trunc_i64_to_i1:
; SI: buffer_load_dwordx2 v{{\[}}[[VLO:[0-9]+]]:{{[0-9]+\]}}
; SI: v_and_b32_e32 [[MASKED:v[0-9]+]], 1, v[[VLO]]
; SI: v_cmp_eq_i32_e32 vcc, 1, [[MASKED]]
; SI: v_cndmask_b32_e64 {{v[0-9]+}}, -12, 63, vcc
define void @v_trunc_i64_to_i1(i32 addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %x = load i64, i64 addrspace(1)* %gep

  %trunc = trunc i64 %x to i1
  %sel = select i1 %trunc, i32 63, i32 -12
  store i32 %sel, i32 addrspace(1)* %out.gep
  ret void
}
