; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s

; Test combine to reduce the width of a 64-bit shift to 32-bit if
; truncated to 16-bit.

; GCN-LABEL: {{^}}trunc_srl_i64_16_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_lshrrev_b32_e32 v0, 16, v0
; GCN-NEXT: s_setpc_b64
define i16 @trunc_srl_i64_16_to_i16(i64 %x) {
  %shift = lshr i64 %x, 16
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_srl_i64_17_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_lshrrev_b64 v[0:1], 17, v[0:1]
; GCN-NEXT: s_setpc_b64
define i16 @trunc_srl_i64_17_to_i16(i64 %x) {
  %shift = lshr i64 %x, 17
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_srl_i55_16_to_i15:
; GCN: s_waitcnt
; GCN-NEXT: v_lshrrev_b32_e32 v0, 15, v0
; GCN-NEXT: v_add_u16_e32 v0, 4, v0
; GCN-NEXT: s_setpc_b64
define i15 @trunc_srl_i55_16_to_i15(i55 %x) {
  %shift = lshr i55 %x, 15
  %trunc = trunc i55 %shift to i15
  %add = add i15 %trunc, 4
  ret i15 %add
}

; GCN-LABEL: {{^}}trunc_sra_i64_16_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_lshrrev_b32_e32 v0, 16, v0
; GCN-NEXT: s_setpc_b64
define i16 @trunc_sra_i64_16_to_i16(i64 %x) {
  %shift = ashr i64 %x, 16
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_sra_i64_17_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_lshrrev_b64 v[0:1], 17, v[0:1]
; GCN-NEXT: s_setpc_b64
define i16 @trunc_sra_i64_17_to_i16(i64 %x) {
  %shift = ashr i64 %x, 17
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_i64_16_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0
; GCN-NEXT: s_setpc_b64
define i16 @trunc_shl_i64_16_to_i16(i64 %x) {
  %shift = shl i64 %x, 16
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_i64_17_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, 0
; GCN-NEXT: s_setpc_b64
define i16 @trunc_shl_i64_17_to_i16(i64 %x) {
  %shift = shl i64 %x, 17
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_srl_v2i64_16_to_v2i16:
; GCN: s_waitcnt
; GCN-DAG: v_lshrrev_b32_e32 v0, 16, v0
; GCN-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff0000
; GCN: v_and_or_b32 v0, v2, [[MASK]], v0
; GCN-NEXT: s_setpc_b64
define <2 x i16> @trunc_srl_v2i64_16_to_v2i16(<2 x i64> %x) {
  %shift = lshr <2 x i64> %x, <i64 16, i64 16>
  %trunc = trunc <2 x i64> %shift to <2 x i16>
  ret <2 x i16> %trunc
}

; GCN-LABEL: {{^}}s_trunc_srl_i64_16_to_i16:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_lshr_b32 [[VAL_SHIFT:s[0-9]+]], [[VAL]], 16
; GCN: s_or_b32 [[RESULT:s[0-9]+]], [[VAL_SHIFT]], 4
; GCN: v_mov_b32_e32 [[V_RESULT:v[0-9]+]], [[RESULT]]
; GCN: global_store_short v{{\[[0-9]+:[0-9]+\]}}, [[V_RESULT]]
define amdgpu_kernel void @s_trunc_srl_i64_16_to_i16(i64 %x) {
  %shift = lshr i64 %x, 16
  %trunc = trunc i64 %shift to i16
  %add = or i16 %trunc, 4
  store i16 %add, i16 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}trunc_srl_i64_var_mask15_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_and_b32_e32 v1, 15, v2
; GCN-NEXT: v_lshrrev_b32_e32 v0, v1, v0
; GCN-NEXT: s_setpc_b64
define i16 @trunc_srl_i64_var_mask15_to_i16(i64 %x, i64 %amt) {
  %amt.masked = and i64 %amt, 15
  %shift = lshr i64 %x, %amt.masked
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_srl_i64_var_mask16_to_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_and_b32_e32 v2, 16, v2
; GCN-NEXT: v_lshrrev_b64 v[0:1], v2, v[0:1]
; GCN-NEXT: s_setpc_b64
define i16 @trunc_srl_i64_var_mask16_to_i16(i64 %x, i64 %amt) {
  %amt.masked = and i64 %amt, 16
  %shift = lshr i64 %x, %amt.masked
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_srl_i64_var_mask31_to_i16:
; GCN: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT: v_and_b32_e32 v2, 31, v2
; GCN-NEXT: v_lshrrev_b64 v[0:1], v2, v[0:1]
; GCN-NEXT: s_setpc_b64 s[30:31]
define i16 @trunc_srl_i64_var_mask31_to_i16(i64 %x, i64 %amt) {
  %amt.masked = and i64 %amt, 31
  %shift = lshr i64 %x, %amt.masked
  %trunc = trunc i64 %shift to i16
  ret i16 %trunc
}
