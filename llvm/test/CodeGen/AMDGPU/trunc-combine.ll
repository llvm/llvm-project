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
; GCN: v_add_u16_e32 v0, 4, v0
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
; GCN: v_add_u16_e32 v0, 4, v0
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

; GCN-LABEL: {{^}}truncate_high_elt_extract_vector:
; GCN: s_load_dword s
; GCN: s_load_dword s
; GCN: s_sext_i32_i16
; GCN: s_sext_i32_i16
; GCN: v_mul_i32_i24
; GCN: v_lshrrev_b32_e32
define amdgpu_kernel void @truncate_high_elt_extract_vector(<2 x i16> addrspace(1)* nocapture readonly %arg, <2 x i16> addrspace(1)* nocapture readonly %arg1, <2 x i16> addrspace(1)* nocapture %arg2) local_unnamed_addr {
bb:
  %tmp = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %arg, i64 undef
  %tmp3 = load <2 x i16>, <2 x i16> addrspace(1)* %tmp, align 4
  %tmp4 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %arg1, i64 undef
  %tmp5 = load <2 x i16>, <2 x i16> addrspace(1)* %tmp4, align 4
  %tmp6 = sext <2 x i16> %tmp3 to <2 x i32>
  %tmp7 = sext <2 x i16> %tmp5 to <2 x i32>
  %tmp8 = extractelement <2 x i32> %tmp6, i64 0
  %tmp9 = extractelement <2 x i32> %tmp7, i64 0
  %tmp10 = mul nsw i32 %tmp9, %tmp8
  %tmp11 = insertelement <2 x i32> undef, i32 %tmp10, i32 0
  %tmp12 = insertelement <2 x i32> %tmp11, i32 undef, i32 1
  %tmp13 = lshr <2 x i32> %tmp12, <i32 16, i32 16>
  %tmp14 = trunc <2 x i32> %tmp13 to <2 x i16>
  %tmp15 = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %arg2, i64 undef
  store <2 x i16> %tmp14, <2 x i16> addrspace(1)* %tmp15, align 4
  ret void
}
