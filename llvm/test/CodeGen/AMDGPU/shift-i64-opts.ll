; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu7.04 < %s | FileCheck -check-prefix=GCN %s


; lshr (i64 x), c: c > 32 => reg_sequence lshr (i32 hi_32(x)), (c - 32), 0
; GCN-LABEL: {{^}}lshr_i64_35:
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 3, v1
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
define i64 @lshr_i64_35(i64 %in) {
  %shl = lshr i64 %in, 35
  ret i64 %shl
}

; GCN-LABEL: {{^}}lshr_i64_63:
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 31, v1
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
define i64 @lshr_i64_63(i64 %in) {
  %shl = lshr i64 %in, 63
  ret i64 %shl
}

; GCN-LABEL: {{^}}lshr_i64_33:
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 1, v1
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
define i64 @lshr_i64_33(i64 %in) {
  %shl = lshr i64 %in, 33
  ret i64 %shl
}

; GCN-LABEL: {{^}}lshr_i64_32:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
define i64 @lshr_i64_32(i64 %in) {
  %shl = lshr i64 %in, 32
  ret i64 %shl
}

; Make sure the and of the constant doesn't prevent bfe from forming
; after 64-bit shift is split.

; GCN-LABEL: {{^}}lshr_and_i64_35:
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfe_u32 v[[BFE:[0-9]+]], v1, 8, 23
define i64 @lshr_and_i64_35(i64 %in) {
  %and = and i64 %in, 9223372036854775807 ; 0x7fffffffffffffff
  %shl = lshr i64 %and, 40
  ret i64 %shl
}

; lshl (i64 x), c: c > 32 => reg_sequence lshl 0, (i32 lo_32(x)), (c - 32)

; GCN-LABEL: {{^}}shl_i64_const_35:
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 3, v0
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
define i64 @shl_i64_const_35(i64 %in) {
  %shl = shl i64 %in, 35
  ret i64 %shl
}

; GCN-LABEL: {{^}}shl_i64_const_32:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
define i64 @shl_i64_const_32(i64 %in) {
  %shl = shl i64 %in, 32
  ret i64 %shl
}

; GCN-LABEL: {{^}}shl_i64_const_63:
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 31, v0
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
define i64 @shl_i64_const_63(i64 %in) {
  %shl = shl i64 %in, 63
  ret i64 %shl
}

; ashr (i64 x), 63 => (ashr lo(x), 31), lo(x)

; GCN-LABEL: {{^}}ashr_i64_const_32:
define i64 @ashr_i64_const_32(i64 %in) {
  %shl = ashr i64 %in, 32
  ret i64 %shl
}

; GCN-LABEL: {{^}}ashr_i64_const_63:
define i64 @ashr_i64_const_63(i64 %in) {
  %shl = ashr i64 %in, 63
  ret i64 %shl
}

; GCN-LABEL: {{^}}trunc_shl_31_i32_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 31, v0
define i32 @trunc_shl_31_i32_i64(i64 %in) {
  %shl = shl i64 %in, 31
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_15_i16_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 15, v0
define i16 @trunc_shl_15_i16_i64(i64 %in) {
  %shl = shl i64 %in, 15
  %trunc = trunc i64 %shl to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_15_i16_i32:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 15, v0
define i16 @trunc_shl_15_i16_i32(i32 %in) {
  %shl = shl i32 %in, 15
  %trunc = trunc i32 %shl to i16
  ret i16 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_7_i8_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 7, v0
define i8 @trunc_shl_7_i8_i64(i64 %in) {
  %shl = shl i64 %in, 7
  %trunc = trunc i64 %shl to i8
  ret i8 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_1_i2_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 1, v0
define i2 @trunc_shl_1_i2_i64(i64 %in) {
  %shl = shl i64 %in, 1
  %trunc = trunc i64 %shl to i2
  ret i2 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_1_i32_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 1, v0
define i32 @trunc_shl_1_i32_i64(i64 %in) {
  %shl = shl i64 %in, 1
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_16_i32_i64:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 16, v0
define i32 @trunc_shl_16_i32_i64(i64 %in) {
  %shl = shl i64 %in, 16
  %trunc = trunc i64 %shl to i32
  ret i32 %trunc
}

; GCN-LABEL: {{^}}trunc_shl_33_i32_i64:
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[ZERO]]
define amdgpu_kernel void @trunc_shl_33_i32_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 33
  %trunc = trunc i64 %shl to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_16_v2i32_v2i64:
; GCN-DAG: v_lshlrev_b32_e32 v[[RESHI:[0-9]+]], 16, v2
; GCN-DAG: v_lshlrev_b32_e32 v[[RESLO:[0-9]+]], 16, v0
define <2 x i32> @trunc_shl_16_v2i32_v2i64(<2 x i64> %in) {
  %shl = shl <2 x i64> %in, <i64 16, i64 16>
  %trunc = trunc <2 x i64> %shl to <2 x i32>
  ret <2 x i32> %trunc
}

; GCN-LABEL: {{^}}trunc_shl_31_i32_i64_multi_use:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshl_b64 v[[[RESLO:[0-9]+]]:[[RESHI:[0-9]+]]], [[VAL]], 31
; GCN: buffer_store_dword v[[RESLO]]
; GCN: buffer_store_dwordx2 v[[[RESLO]]:[[RESHI]]]
define amdgpu_kernel void @trunc_shl_31_i32_i64_multi_use(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.tid = getelementptr inbounds i64, ptr addrspace(1) %in, i32 %tid
  %val = load i64, ptr addrspace(1) %in.tid
  %shl = shl i64 %val, 31
  %trunc = trunc i64 %shl to i32
  store volatile i32 %trunc, ptr addrspace(1) %out
  store volatile i64 %shl, ptr addrspace(1) %in.tid
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_and31:
; GCN:     v_lshlrev_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GCN-NOT: v_lshl_b64
; GCN-NOT: v_lshlrev_b64
define i32 @trunc_shl_and31(i64 %arg, i32 inreg %arg2) {
  %tmp3 = and i32 %arg2, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %arg, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  ret i32 %tmp6
}

; GCN-LABEL: {{^}}trunc_shl_and30:
; GCN:     s_and_b32 s[[AMT:[0-9]+]], s{{[0-9]+}}, 30
; GCN:     v_lshlrev_b32_e32 v{{[0-9]+}}, s[[AMT]], v{{[0-9]+}}
; GCN-NOT: v_lshl_b64
; GCN-NOT: v_lshlrev_b64
define i32 @trunc_shl_and30(i64 %arg, i32 inreg %arg2) {
  %tmp3 = and i32 %arg2, 30
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %arg, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  ret i32 %tmp6
}

; GCN-LABEL: {{^}}trunc_shl_wrong_and63:
; Negative test, wrong constant
; GCN: v_lshl_b64
define i32 @trunc_shl_wrong_and63(i64 %arg, i32 inreg %arg2) {
  %tmp3 = and i32 %arg2, 63
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %arg, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  ret i32 %tmp6
}

; GCN-LABEL: {{^}}trunc_shl_no_and:
; Negative test, shift can be full 64 bit
; GCN: v_lshl_b64
define i32 @trunc_shl_no_and(i64 %arg, i32 inreg %arg2) {
  %tmp4 = zext i32 %arg2 to i64
  %tmp5 = shl i64 %arg, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  ret i32 %tmp6
}

; GCN-LABEL: {{^}}trunc_shl_vec_vec:
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 3
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 5
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 6
define <4 x i64> @trunc_shl_vec_vec(<4 x i64> %v) {
  %shl = shl <4 x i64> %v, <i64 3, i64 4, i64 5, i64 6>
  ret <4 x i64> %shl
}
