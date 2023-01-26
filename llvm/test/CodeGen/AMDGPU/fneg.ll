; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: not llc -march=r600 -mcpu=redwood < %s | FileCheck -enable-var-scope -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}s_fneg_f32:
; R600: -PV

; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: s_xor_b32 [[NEG_VAL:s[0-9]+]], [[VAL]], 0x80000000
; GCN: v_mov_b32_e32 v{{[0-9]+}}, [[NEG_VAL]]
define amdgpu_kernel void @s_fneg_f32(ptr addrspace(1) %out, float %in) {
  %fneg = fsub float -0.000000e+00, %in
  store float %fneg, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_fneg_v2f32:
; R600: -PV
; R600: -PV

; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
define amdgpu_kernel void @s_fneg_v2f32(ptr addrspace(1) nocapture %out, <2 x float> %in) {
  %fneg = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %in
  store <2 x float> %fneg, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_fneg_v4f32:
; R600: -PV
; R600: -T
; R600: -PV
; R600: -PV

; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
; GCN: s_xor_b32 {{s[0-9]+}}, {{s[0-9]+}}, 0x80000000
define amdgpu_kernel void @s_fneg_v4f32(ptr addrspace(1) nocapture %out, <4 x float> %in) {
  %fneg = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %in
  store <4 x float> %fneg, ptr addrspace(1) %out
  ret void
}

; DAGCombiner will transform:
; (fneg (f32 bitcast (i32 a))) => (f32 bitcast (xor (i32 a), 0x80000000))
; unless the target returns true for isNegFree()

; FUNC-LABEL: {{^}}fsub0_f32:

; GCN: v_sub_f32_e64 v{{[0-9]}}, 0, s{{[0-9]+$}}

; R600-NOT: XOR
; R600: -KC0[2].Z
define amdgpu_kernel void @fsub0_f32(ptr addrspace(1) %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fsub = fsub float 0.0, %bc
  store float %fsub, ptr addrspace(1) %out
  ret void
}
; FUNC-LABEL: {{^}}fneg_free_f32:
; SI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c

; GCN: s_xor_b32 [[RES:s[0-9]+]], [[NEG_VALUE]], 0x80000000
; GCN: v_mov_b32_e32 [[V_RES:v[0-9]+]], [[RES]]
; GCN: buffer_store_dword [[V_RES]]

; R600-NOT: XOR
; R600: -PV.W
define amdgpu_kernel void @fneg_free_f32(ptr addrspace(1) %out, i32 %in) {
  %bc = bitcast i32 %in to float
  %fsub = fsub float -0.0, %bc
  store float %fsub, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fneg_fold_f32:
; SI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[NEG_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN-NOT: xor
; GCN: v_mul_f32_e64 v{{[0-9]+}}, -[[NEG_VALUE]], [[NEG_VALUE]]
define amdgpu_kernel void @fneg_fold_f32(ptr addrspace(1) %out, float %in) {
  %fsub = fsub float -0.0, %in
  %fmul = fmul float %fsub, %in
  store float %fmul, ptr addrspace(1) %out
  ret void
}

; Make sure we turn some integer operations back into fabs
; FUNC-LABEL: {{^}}bitpreserve_fneg_f32:
; GCN: v_mul_f32_e64 v{{[0-9]+}}, s{{[0-9]+}}, -4.0
define amdgpu_kernel void @bitpreserve_fneg_f32(ptr addrspace(1) %out, float %in) {
  %in.bc = bitcast float %in to i32
  %int.abs = xor i32 %in.bc, 2147483648
  %bc = bitcast i32 %int.abs to float
  %fadd = fmul float %bc, 4.0
  store float %fadd, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_fneg_i32:
; GCN: s_load_dword [[IN:s[0-9]+]]
; GCN: s_xor_b32 [[FNEG:s[0-9]+]], [[IN]], 0x80000000
; GCN: v_mov_b32_e32 [[V_FNEG:v[0-9]+]], [[FNEG]]
define amdgpu_kernel void @s_fneg_i32(ptr addrspace(1) %out, i32 %in) {
  %fneg = xor i32 %in, -2147483648
  store i32 %fneg, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_i32:
; GCN: s_waitcnt
; GCN-NEXT: v_xor_b32_e32 v0, 0x80000000, v0
; GCN-NEXT: s_setpc_b64
define i32 @v_fneg_i32(i32 %in) {
  %fneg = xor i32 %in, -2147483648
  ret i32 %fneg
}

; FUNC-LABEL: {{^}}s_fneg_i32_fp_use:
; GCN: s_load_dword [[IN:s[0-9]+]]
; GCN: s_xor_b32 [[FNEG:s[0-9]+]], [[IN]], 0x80000000
; GCN: v_add_f32_e64 v{{[0-9]+}}, [[FNEG]], 2.0
define amdgpu_kernel void @s_fneg_i32_fp_use(ptr addrspace(1) %out, i32 %in) {
  %fneg = xor i32 %in, -2147483648
  %bitcast = bitcast i32 %fneg to float
  %fadd = fadd float %bitcast, 2.0
  store float %fadd, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_i32_fp_use:
; GCN: s_waitcnt
; GCN-NEXT: v_xor_b32_e32 v0, 0x80000000, v0
; GCN-NEXT: v_add_f32_e32 v0, 2.0, v0
; GCN-NEXT: s_setpc_b64
define float @v_fneg_i32_fp_use(i32 %in) {
  %fneg = xor i32 %in, -2147483648
  %bitcast = bitcast i32 %fneg to float
  %fadd = fadd float %bitcast, 2.0
  ret float %fadd
}

; FUNC-LABEL: {{^}}s_fneg_i64:
; GCN: s_xor_b32 s[[NEG_HI:[0-9]+]], s{{[0-9]+}}, 0x80000000
define amdgpu_kernel void @s_fneg_i64(ptr addrspace(1) %out, i64 %in) {
  %fneg = xor i64 %in, -9223372036854775808
  store i64 %fneg, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_i64:
; GCN: s_waitcnt
; GCN-NEXT: v_xor_b32_e32 v1, 0x80000000, v1
; GCN-NEXT: s_setpc_b64
define i64 @v_fneg_i64(i64 %in) {
  %fneg = xor i64 %in, -9223372036854775808
  ret i64 %fneg
}

; FUNC-LABEL: {{^}}s_fneg_i64_fp_use:
; GCN: s_xor_b32 s[[NEG_HI:[0-9]+]], s{{[0-9]+}}, 0x80000000
; GCN: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 2.0
define amdgpu_kernel void @s_fneg_i64_fp_use(ptr addrspace(1) %out, i64 %in) {
  %fneg = xor i64 %in, -9223372036854775808
  %bitcast = bitcast i64 %fneg to double
  %fadd = fadd double %bitcast, 2.0
  store double %fadd, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_i64_fp_use:
; GCN: s_waitcnt
; GCN-NEXT: v_xor_b32_e32 v1, 0x80000000, v1
; GCN-NEXT: v_add_f64 v[0:1], v[0:1], 2.0
; GCN-NEXT: s_setpc_b64
define double @v_fneg_i64_fp_use(i64 %in) {
  %fneg = xor i64 %in, -9223372036854775808
  %bitcast = bitcast i64 %fneg to double
  %fadd = fadd double %bitcast, 2.0
  ret double %fadd
}

; FUNC-LABEL: {{^}}v_fneg_i16:
; GCN: s_waitcnt
; GCN-NEXT: v_xor_b32_e32 v0, 0xffff8000, v0
; GCN-NEXT: s_setpc_b64
define i16 @v_fneg_i16(i16 %in) {
  %fneg = xor i16 %in, -32768
  ret i16 %fneg
}

; FUNC-LABEL: {{^}}s_fneg_i16_fp_use:
; SI: v_cvt_f32_f16_e64 [[CVT0:v[0-9]+]], -s{{[0-9]+}}
; SI: v_add_f32_e32 [[ADD:v[0-9]+]], 2.0, [[CVT0]]
; SI: v_cvt_f16_f32_e32 [[CVT1:v[0-9]+]], [[ADD]]

; VI: s_load_dword [[IN:s[0-9]+]]
; VI: v_mov_b32_e32 [[K:v[0-9]+]], 0xffff8000
; VI: v_xor_b32_e32 [[NEG:v[0-9]+]], [[IN]], [[K]]
; VI: v_add_f16_e32 v{{[0-9]+}}, 2.0, [[NEG]]
define amdgpu_kernel void @s_fneg_i16_fp_use(ptr addrspace(1) %out, i16 %in) {
  %fneg = xor i16 %in, -32768
  %bitcast = bitcast i16 %fneg to half
  %fadd = fadd half %bitcast, 2.0
  store half %fadd, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_i16_fp_use:
; SI: s_waitcnt
; SI-NEXT: v_cvt_f32_f16_e64 v0, -v0
; SI-NEXT: v_add_f32_e32 v0, 2.0, v0
; SI-NEXT: s_setpc_b64

; VI: s_waitcnt
; VI-NEXT: v_xor_b32_e32 v0, 0xffff8000, v0
; VI-NEXT: v_add_f16_e32 v0, 2.0, v0
; VI-NEXT: s_setpc_b64
define half @v_fneg_i16_fp_use(i16 %in) {
  %fneg = xor i16 %in, -32768
  %bitcast = bitcast i16 %fneg to half
  %fadd = fadd half %bitcast, 2.0
  ret half %fadd
}

; FIXME: This is terrible
; FUNC-LABEL: {{^}}s_fneg_v2i16:
; SI: s_and_b32 s5, s4, 0xffff0000
; SI: s_xor_b32 s4, s4, 0x8000
; SI: s_and_b32 s4, s4, 0xffff
; SI: s_or_b32 s4, s4, s5
; SI: s_add_i32 s4, s4, 0x80000000

; VI: s_lshr_b32 s5, s4, 16
; VI: s_xor_b32 s4, s4, 0x8000
; VI: s_xor_b32 s5, s5, 0x8000
; VI: s_and_b32 s4, s4, 0xffff
; VI: s_lshl_b32 s5, s5, 16
; VI: s_or_b32 s4, s4, s5
define amdgpu_kernel void @s_fneg_v2i16(ptr addrspace(1) %out, i32 %arg) {
  %in = bitcast i32 %arg to <2 x i16>
  %fneg = xor <2 x i16> %in, <i16 -32768, i16 -32768>
  store <2 x i16> %fneg, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_v2i16:
; SI: v_lshlrev_b32_e32 v1, 16, v1
; SI: v_xor_b32_e32 v0, 0x8000, v0
; SI: v_xor_b32_e32 v1, 0x80000000, v1
; SI: v_and_b32_e32 v0, 0xffff, v0
; SI: v_or_b32_e32 v0, v0, v1
; SI: v_lshrrev_b32_e32 v1, 16, v1

; VI: s_waitcnt
; VI-NEXT: v_xor_b32_e32 v0, 0x80008000, v0
; VI-NEXT: s_setpc_b64
define <2 x i16> @v_fneg_v2i16(<2 x i16> %in) {
  %fneg = xor <2 x i16> %in, <i16 -32768, i16 -32768>
  ret <2 x i16> %fneg
}

; FUNC-LABEL: {{^}}s_fneg_v2i16_fp_use:
; SI: s_lshr_b32 s3, s2, 16
; SI: v_cvt_f32_f16_e64 v0, -s3
; SI: v_cvt_f32_f16_e64 v1, -s2

; VI: s_lshr_b32 s5, s4, 16
; VI: s_xor_b32 s5, s5, 0x8000
; VI: s_xor_b32 s4, s4, 0x8000
; VI: v_mov_b32_e32 v0, s5
; VI: v_add_f16_sdwa v0, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_add_f16_e64 v1, s4, 2.0
; VI: v_or_b32_e32 v0, v1, v0
define amdgpu_kernel void @s_fneg_v2i16_fp_use(ptr addrspace(1) %out, i32 %arg) {
  %in = bitcast i32 %arg to <2 x i16>
  %fneg = xor <2 x i16> %in, <i16 -32768, i16 -32768>
  %bitcast = bitcast <2 x i16> %fneg to <2 x half>
  %fadd = fadd <2 x half> %bitcast, <half 2.0, half 2.0>
  store <2 x half> %fadd, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_v2i16_fp_use:
; SI: v_lshrrev_b32_e32 v1, 16, v0
; SI: v_cvt_f32_f16_e64 v0, -v0
; SI: v_cvt_f32_f16_e64 v1, -v1
; SI: v_add_f32_e32 v0, 2.0, v0
; SI: v_add_f32_e32 v1, 2.0, v1

; VI: s_waitcnt
; VI: v_xor_b32_e32 v0, 0x80008000, v0
; VI: v_mov_b32_e32 v1, 0x4000
; VI: v_add_f16_sdwa v1, v0, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_add_f16_e32 v0, 2.0, v0
; VI: v_or_b32_e32 v0, v0, v1
; VI: s_setpc_b64
define <2 x half> @v_fneg_v2i16_fp_use(i32 %arg) {
  %in = bitcast i32 %arg to <2 x i16>
  %fneg = xor <2 x i16> %in, <i16 -32768, i16 -32768>
  %bitcast = bitcast <2 x i16> %fneg to <2 x half>
  %fadd = fadd <2 x half> %bitcast, <half 2.0, half 2.0>
  ret <2 x half> %fadd
}
