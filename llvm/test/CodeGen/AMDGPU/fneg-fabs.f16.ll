; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=CI,GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck --check-prefixes=VI,GFX89,GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GFX89,GFX9,GCN %s

; GCN-LABEL: {{^}}fneg_fabs_fadd_f16:
; CI-DAG: v_cvt_f32_f16_e32
; CI-DAG: v_cvt_f32_f16_e64 [[CVT_ABS_X:v[0-9]+]], |s{{[0-9]+}}|
; CI: v_sub_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[CVT_ABS_X]]

; GFX89-NOT: _and
; GFX89: v_sub_f16_e64 {{v[0-9]+}}, {{s[0-9]+}}, |{{v[0-9]+}}|
define amdgpu_kernel void @fneg_fabs_fadd_f16(half addrspace(1)* %out, half %x, half %y) {
  %fabs = call half @llvm.fabs.f16(half %x)
  %fsub = fsub half -0.0, %fabs
  %fadd = fadd half %y, %fsub
  store half %fadd, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_fmul_f16:
; CI-DAG: v_cvt_f32_f16_e32
; CI-DAG: v_cvt_f32_f16_e64 [[CVT_NEG_ABS_X:v[0-9]+]], -|{{s[0-9]+}}|
; CI: v_mul_f32_e32 {{v[0-9]+}},  {{v[0-9]+}}, [[CVT_NEG_ABS_X]]
; CI: v_cvt_f16_f32_e32

; GFX89-NOT: _and
; GFX89: v_mul_f16_e64 [[MUL:v[0-9]+]], {{s[0-9]+}}, -|{{v[0-9]+}}|
; GFX89-NOT: [[MUL]]
; GFX89: {{flat|global}}_store_short v{{.+}}, [[MUL]]
define amdgpu_kernel void @fneg_fabs_fmul_f16(half addrspace(1)* %out, half %x, half %y) {
  %fabs = call half @llvm.fabs.f16(half %x)
  %fsub = fsub half -0.0, %fabs
  %fmul = fmul half %y, %fsub
  store half %fmul, half addrspace(1)* %out, align 2
  ret void
}

; DAGCombiner will transform:
; (fabs (f16 bitcast (i16 a))) => (f16 bitcast (and (i16 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; GCN-LABEL: {{^}}fneg_fabs_free_f16:
; GCN: {{s_or_b32 s[0-9]+, s[0-9]+, 0x8000|s_bitset1_b32 s[0-9]+, 15}}
define amdgpu_kernel void @fneg_fabs_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc = bitcast i16 %in to half
  %fabs = call half @llvm.fabs.f16(half %bc)
  %fsub = fsub half -0.0, %fabs
  store half %fsub, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_f16:
; GCN: {{s_or_b32 s[0-9]+, s[0-9]+, 0x8000|s_bitset1_b32 s[0-9]+, 15}}
define amdgpu_kernel void @fneg_fabs_f16(half addrspace(1)* %out, half %in) {
  %fabs = call half @llvm.fabs.f16(half %in)
  %fsub = fsub half -0.0, %fabs
  store half %fsub, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fabs_f16:
; GCN: v_or_b32_e32 v{{[0-9]+}}, 0x8000, v{{[0-9]+}}
define amdgpu_kernel void @v_fneg_fabs_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %val = load half, half addrspace(1)* %in, align 2
  %fabs = call half @llvm.fabs.f16(half %val)
  %fsub = fsub half -0.0, %fabs
  store half %fsub, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}s_fneg_fabs_v2f16_non_bc_src:
; GFX9-DAG: s_load_dword [[VAL:s[0-9]+]]
; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x40003c00
; GFX9: v_pk_add_f16 [[ADD:v[0-9]+]], [[VAL]], [[K]]
; GFX9: v_or_b32_e32 [[RESULT:v[0-9]+]], 0x80008000, [[ADD]]

; VI: v_or_b32_e32 v{{[0-9]+}}, 0x80008000, v{{[0-9]+}}
define amdgpu_kernel void @s_fneg_fabs_v2f16_non_bc_src(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %add = fadd <2 x half> %in, <half 1.0, half 2.0>
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %add)
  %fneg.fabs = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  store <2 x half> %fneg.fabs, <2 x half> addrspace(1)* %out
  ret void
}

; FIXME: single bit op

; Combine turns this into integer op when bitcast source (from load)

; GCN-LABEL: {{^}}s_fneg_fabs_v2f16_bc_src:
; GCN: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80008000
define amdgpu_kernel void @s_fneg_fabs_v2f16_bc_src(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  %fneg.fabs = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  store <2 x half> %fneg.fabs, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_v4f16:
; GCN: s_mov_b32 [[MASK:s[0-9]+]], 0x80008000
; GCN: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, [[MASK]]
; GCN: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, [[MASK]]
; GCN: {{flat|global}}_store_dwordx2
define amdgpu_kernel void @fneg_fabs_v4f16(<4 x half> addrspace(1)* %out, <4 x half> %in) {
  %fabs = call <4 x half> @llvm.fabs.v4f16(<4 x half> %in)
  %fsub = fsub <4 x half> <half -0.0, half -0.0, half -0.0, half -0.0>, %fabs
  store <4 x half> %fsub, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fold_user_fneg_fabs_v2f16:
; CI: s_load_dword [[IN:s[0-9]+]]
; CI: s_lshr_b32
; CI: v_cvt_f32_f16_e64 v{{[0-9]+}}, |s{{[0-9]+}}|
; CI: v_cvt_f32_f16_e64 v{{[0-9]+}}, |s{{[0-9]+}}|
; CI: v_mul_f32_e32 v{{[0-9]+}}, -4.0, v{{[0-9]+}}
; CI: v_mul_f32_e32 v{{[0-9]+}}, -4.0, v{{[0-9]+}}

; VI: v_mul_f16_e64 v{{[0-9]+}}, |s{{[0-9]+}}|, -4.0
; VI: v_mul_f16_sdwa v{{[0-9]+}}, |v{{[0-9]+}}|, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD

; GFX9: s_and_b32 [[ABS:s[0-9]+]], s{{[0-9]+}}, 0x7fff7fff
; GFX9: v_pk_mul_f16 v{{[0-9]+}}, [[ABS]], -4.0 op_sel_hi:[1,0]
define amdgpu_kernel void @fold_user_fneg_fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in) #0 {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  %fneg.fabs = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  %mul = fmul <2 x half> %fneg.fabs, <half 4.0, half 4.0>
  store <2 x half> %mul, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fneg_multi_use_fabs_v2f16:
; GFX9: s_and_b32 [[ABS:s[0-9]+]], s{{[0-9]+}}, 0x7fff7fff
; GFX9: s_xor_b32 [[NEG:s[0-9]+]], [[ABS]], 0x80008000
; GFX9: v_mov_b32_e32 [[V_ABS:v[0-9]+]], [[ABS]]
; GFX9-DAG: v_mov_b32_e32 [[V_NEG:v[0-9]+]], [[NEG]]
; GFX9-DAG: global_store_dword v{{[0-9]+}}, [[V_ABS]], s{{\[[0-9]+:[0-9]+\]}}
; GFX9: global_store_dword v{{[0-9]+}}, [[V_NEG]], s{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @s_fneg_multi_use_fabs_v2f16(<2 x half> addrspace(1)* %out0, <2 x half> addrspace(1)* %out1, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  %fneg = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out0
  store <2 x half> %fneg, <2 x half> addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}s_fneg_multi_use_fabs_foldable_neg_v2f16:
; GFX9: s_and_b32 [[ABS:s[0-9]+]], s{{[0-9]+}}, 0x7fff7fff
; GFX9: v_pk_mul_f16 v{{[0-9]+}}, [[ABS]], -4.0 op_sel_hi:[1,0]
define amdgpu_kernel void @s_fneg_multi_use_fabs_foldable_neg_v2f16(<2 x half> addrspace(1)* %out0, <2 x half> addrspace(1)* %out1, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  %fneg = fsub <2 x half> <half -0.0, half -0.0>, %fabs
  %mul = fmul <2 x half> %fneg, <half 4.0, half 4.0>
  store <2 x half> %fabs, <2 x half> addrspace(1)* %out0
  store <2 x half> %mul, <2 x half> addrspace(1)* %out1
  ret void
}

declare half @llvm.fabs.f16(half) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1
declare <4 x half> @llvm.fabs.v4f16(<4 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
