; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=GCN %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=GCN %s


; lshr (i64 x), c: c > 32 => reg_sequence lshr (i32 hi_32(x)), (c - 32), 0
; GCN-LABEL: {{^}}lshr_i64_35:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 3, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @lshr_i64_35(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = lshr i64 %val, 35
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_63:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 31, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @lshr_i64_63(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = lshr i64 %val, 63
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_33:
; GCN-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; GCN-DAG: v_lshrrev_b32_e32 v[[LO:[0-9]+]], 1, [[VAL]]
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @lshr_i64_33(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = lshr i64 %val, 33
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}lshr_i64_32:
; GCN-DAG: buffer_load_dword v[[LO:[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @lshr_i64_32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = lshr i64 %val, 32
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; Make sure the and of the constant doesn't prevent bfe from forming
; after 64-bit shift is split.

; GCN-LABEL: {{^}}lshr_and_i64_35:
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN-DAG: buffer_load_dword v[[LO:[0-9]+]]
; GCN: v_bfe_u32 v[[BFE:[0-9]+]], v[[LO]], 8, 23
; GCN: buffer_store_dwordx2 v[[[BFE]]:[[ZERO]]]
define amdgpu_kernel void @lshr_and_i64_35(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %and = and i64 %val, 9223372036854775807 ; 0x7fffffffffffffff
  %shl = lshr i64 %and, 40
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; lshl (i64 x), c: c > 32 => reg_sequence lshl 0, (i32 lo_32(x)), (c - 32)

; GCN-LABEL: {{^}}shl_i64_const_35:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 3, [[VAL]]
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @shl_i64_const_35(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 35
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}shl_i64_const_32:
; GCN-DAG: buffer_load_dword v[[HI:[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @shl_i64_const_32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 32
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}shl_i64_const_63:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 v[[HI:[0-9]+]], 31, [[VAL]]
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LO]]:[[HI]]]
define amdgpu_kernel void @shl_i64_const_63(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 63
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; ashr (i64 x), 63 => (ashr lo(x), 31), lo(x)

; GCN-LABEL: {{^}}ashr_i64_const_32:
define amdgpu_kernel void @ashr_i64_const_32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = ashr i64 %val, 32
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}ashr_i64_const_63:
define amdgpu_kernel void @ashr_i64_const_63(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = ashr i64 %val, 63
  store i64 %shl, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_31_i32_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 31, [[VAL]]
; GCN: buffer_store_dword [[SHL]]
define amdgpu_kernel void @trunc_shl_31_i32_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 31
  %trunc = trunc i64 %shl to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_15_i16_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 15, [[VAL]]
; GCN: buffer_store_short [[SHL]]
define amdgpu_kernel void @trunc_shl_15_i16_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 15
  %trunc = trunc i64 %shl to i16
  store i16 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_15_i16_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 15, [[VAL]]
; GCN: buffer_store_short [[SHL]]
define amdgpu_kernel void @trunc_shl_15_i16_i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i32, ptr addrspace(1) %in
  %shl = shl i32 %val, 15
  %trunc = trunc i32 %shl to i16
  store i16 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_7_i8_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 7, [[VAL]]
; GCN: buffer_store_byte [[SHL]]
define amdgpu_kernel void @trunc_shl_7_i8_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 7
  %trunc = trunc i64 %shl to i8
  store i8 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_1_i2_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 1, [[VAL]]
; GCN: v_and_b32_e32 [[AND:v[0-9]+]], 2, [[SHL]]
; GCN: buffer_store_byte [[AND]]
define amdgpu_kernel void @trunc_shl_1_i2_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 1
  %trunc = trunc i64 %shl to i2
  store i2 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_1_i32_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 1, [[VAL]]
; GCN: buffer_store_dword [[SHL]]
define amdgpu_kernel void @trunc_shl_1_i32_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 1
  %trunc = trunc i64 %shl to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_16_i32_i64:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 16, [[VAL]]
; GCN: buffer_store_dword [[SHL]]
define amdgpu_kernel void @trunc_shl_16_i32_i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 16
  %trunc = trunc i64 %shl to i32
  store i32 %trunc, ptr addrspace(1) %out
  ret void
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
; GCN: buffer_load_dwordx4 v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; GCN-DAG: v_lshlrev_b32_e32 v[[RESHI:[0-9]+]], 16, v{{[0-9]+}}
; GCN-DAG: v_lshlrev_b32_e32 v[[RESLO:[0-9]+]], 16, v[[LO]]
; GCN: buffer_store_dwordx2 v[[[RESLO]]:[[RESHI]]]
define amdgpu_kernel void @trunc_shl_16_v2i32_v2i64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load <2 x i64>, ptr addrspace(1) %in
  %shl = shl <2 x i64> %val, <i64 16, i64 16>
  %trunc = trunc <2 x i64> %shl to <2 x i32>
  store <2 x i32> %trunc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_31_i32_i64_multi_use:
; GCN: buffer_load_dwordx2 [[VAL:v\[[0-9]+:[0-9]+\]]]
; GCN: v_lshl_b64 v[[[RESLO:[0-9]+]]:[[RESHI:[0-9]+]]], [[VAL]], 31
; GCN: buffer_store_dword v[[RESLO]]
; GCN: buffer_store_dwordx2 v[[[RESLO]]:[[RESHI]]]
define amdgpu_kernel void @trunc_shl_31_i32_i64_multi_use(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %val = load i64, ptr addrspace(1) %in
  %shl = shl i64 %val, 31
  %trunc = trunc i64 %shl to i32
  store volatile i32 %trunc, ptr addrspace(1) %out
  store volatile i64 %shl, ptr addrspace(1) %in
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_and31:
; GCN:     v_lshlrev_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; GCN-NOT: v_lshl_b64
; GCN-NOT: v_lshlrev_b64
define amdgpu_kernel void @trunc_shl_and31(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 31
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_and30:
; GCN:     s_and_b32 s[[AMT:[0-9]+]], s{{[0-9]+}}, 30
; GCN:     v_lshlrev_b32_e32 v{{[0-9]+}}, s[[AMT]], v{{[0-9]+}}
; GCN-NOT: v_lshl_b64
; GCN-NOT: v_lshlrev_b64
define amdgpu_kernel void @trunc_shl_and30(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 30
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_wrong_and63:
; Negative test, wrong constant
; GCN: v_lshl_b64
define amdgpu_kernel void @trunc_shl_wrong_and63(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp3 = and i32 %arg2, 63
  %tmp4 = zext i32 %tmp3 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_no_and:
; Negative test, shift can be full 64 bit
; GCN: v_lshl_b64
define amdgpu_kernel void @trunc_shl_no_and(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(1) nocapture %arg1, i32 %arg2) {
bb:
  %tmp = load i64, ptr addrspace(1) %arg, align 8
  %tmp4 = zext i32 %arg2 to i64
  %tmp5 = shl i64 %tmp, %tmp4
  %tmp6 = trunc i64 %tmp5 to i32
  store i32 %tmp6, ptr addrspace(1) %arg1, align 4
  ret void
}

; GCN-LABEL: {{^}}trunc_shl_vec_vec:
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 3
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 4
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 5
; GCN-DAG: v_lshl_b64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], 6
define amdgpu_kernel void @trunc_shl_vec_vec(ptr addrspace(1) %arg) {
bb:
  %v = load <4 x i64>, ptr addrspace(1) %arg, align 32
  %shl = shl <4 x i64> %v, <i64 3, i64 4, i64 5, i64 6>
  store <4 x i64> %shl, ptr addrspace(1) %arg, align 32
  ret void
}
