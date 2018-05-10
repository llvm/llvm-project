; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs< %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; Make sure high constant 0 isn't pointlessly materialized
; GCN-LABEL: {{^}}trunc_bitcast_i64_lshr_32_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, v1
; GCN-NEXT: s_setpc_b64
define i16 @trunc_bitcast_i64_lshr_32_i16(i64 %bar) {
  %srl = lshr i64 %bar, 32
  %trunc = trunc i64 %srl to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_bitcast_i64_lshr_32_i32:
; GCN: s_waitcnt
; GCN-NEXT: v_mov_b32_e32 v0, v1
; GCN-NEXT: s_setpc_b64
define i32 @trunc_bitcast_i64_lshr_32_i32(i64 %bar) {
  %srl = lshr i64 %bar, 32
  %trunc = trunc i64 %srl to i32
  ret i32 %trunc
}

; GCN-LABEL: {{^}}trunc_bitcast_v2i32_to_i16:
; GCN: _load_dword
; GCN-NOT: _load_dword
; GCN-NOT: v_mov_b32
; GCN: v_add_u32_e32 v0, vcc, 4, v0
define i16 @trunc_bitcast_v2i32_to_i16(<2 x i32> %bar) {
  %load0 = load i32, i32 addrspace(1)* undef
  %load1 = load i32, i32 addrspace(1)* null
  %insert.0 = insertelement <2 x i32> undef, i32 %load0, i32 0
  %insert.1 = insertelement <2 x i32> %insert.0, i32 99, i32 1
  %bc = bitcast <2 x i32> %insert.1 to i64
  %trunc = trunc i64 %bc to i16
  %add = add i16 %trunc, 4
  ret i16 %add
}

; Make sure there's no crash if the source vector type is FP
; GCN-LABEL: {{^}}trunc_bitcast_v2f32_to_i16:
; GCN: _load_dword
; GCN-NOT: _load_dword
; GCN-NOT: v_mov_b32
; GCN: v_add_u32_e32 v0, vcc, 4, v0
define i16 @trunc_bitcast_v2f32_to_i16(<2 x float> %bar) {
  %load0 = load float, float addrspace(1)* undef
  %load1 = load float, float addrspace(1)* null
  %insert.0 = insertelement <2 x float> undef, float %load0, i32 0
  %insert.1 = insertelement <2 x float> %insert.0, float 4.0, i32 1
  %bc = bitcast <2 x float> %insert.1 to i64
  %trunc = trunc i64 %bc to i16
  %add = add i16 %trunc, 4
  ret i16 %add
}
