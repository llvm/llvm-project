; RUN: llc -verify-machineinstrs -mtriple=amdgcn-amd-amdhsa -mcpu=kaveri -mattr=-flat-for-global,+max-private-element-size-16 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI,GCN-NO-TONGA %s
; RUN: llc -verify-machineinstrs -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -mattr=-flat-for-global -mattr=+max-private-element-size-16 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GCN-TONGA %s

; FIXME: Broken on evergreen
; FIXME: For some reason the 8 and 16 vectors are being stored as
; individual elements instead of 128-bit stores.


; FIXME: Why is the constant moved into the intermediate register and
; not just directly into the vector component?

; GCN-LABEL: {{^}}insertelement_v4f32_0:
; GCN: s_load_dwordx4
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: s_mov_b32 [[CONSTREG:s[0-9]+]], 0x40a00000
; GCN-DAG: v_mov_b32_e32 v[[LOW_REG:[0-9]+]], [[CONSTREG]]
; GCN: buffer_store_dwordx4 v{{\[}}[[LOW_REG]]:
define amdgpu_kernel void @insertelement_v4f32_0(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 0
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v4f32_1:
define amdgpu_kernel void @insertelement_v4f32_1(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 1
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v4f32_2:
define amdgpu_kernel void @insertelement_v4f32_2(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 2
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v4f32_3:
define amdgpu_kernel void @insertelement_v4f32_3(<4 x float> addrspace(1)* %out, <4 x float> %a) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 3
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v4i32_0:
define amdgpu_kernel void @insertelement_v4i32_0(<4 x i32> addrspace(1)* %out, <4 x i32> %a) nounwind {
  %vecins = insertelement <4 x i32> %a, i32 999, i32 0
  store <4 x i32> %vecins, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v3f32_1:
define amdgpu_kernel void @insertelement_v3f32_1(<3 x float> addrspace(1)* %out, <3 x float> %a) nounwind {
  %vecins = insertelement <3 x float> %a, float 5.000000e+00, i32 1
  store <3 x float> %vecins, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v3f32_2:
define amdgpu_kernel void @insertelement_v3f32_2(<3 x float> addrspace(1)* %out, <3 x float> %a) nounwind {
  %vecins = insertelement <3 x float> %a, float 5.000000e+00, i32 2
  store <3 x float> %vecins, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_v3f32_3:
define amdgpu_kernel void @insertelement_v3f32_3(<3 x float> addrspace(1)* %out, <3 x float> %a) nounwind {
  %vecins = insertelement <3 x float> %a, float 5.000000e+00, i32 3
  store <3 x float> %vecins, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}insertelement_to_sgpr:
; GCN-NOT: v_readfirstlane
define <4 x float> @insertelement_to_sgpr() nounwind {
  %tmp = load <4 x i32>, <4 x i32> addrspace(4)* undef
  %tmp1 = insertelement <4 x i32> %tmp, i32 0, i32 0
  %tmp2 = call <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32 1, float undef, float undef, <8 x i32> undef, <4 x i32> %tmp1, i1 0, i32 0, i32 0)
  ret <4 x float> %tmp2
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2f32:
; GCN-DAG: v_mov_b32_e32 [[CONST:v[0-9]+]], 0x40a00000
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[LOW_RESULT_REG:[0-9]+]], [[CONST]], v{{[0-9]+}}, [[CC1]]
; GCN: buffer_store_dwordx2 {{v\[}}[[LOW_RESULT_REG]]:
define amdgpu_kernel void @dynamic_insertelement_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x float> %a, float 5.000000e+00, i32 %b
  store <2 x float> %vecins, <2 x float> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v3f32:
; GCN-DAG: v_mov_b32_e32 [[CONST:v[0-9]+]], 0x40a00000
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX:s[0-9]+]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC1]]
; GCN-DAG: buffer_store_dwordx3 v
define amdgpu_kernel void @dynamic_insertelement_v3f32(<3 x float> addrspace(1)* %out, <3 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <3 x float> %a, float 5.000000e+00, i32 %b
  store <3 x float> %vecins, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v4f32:
; GCN-DAG: v_mov_b32_e32 [[CONST:v[0-9]+]], 0x40a00000
; GCN-DAG: v_cmp_ne_u32_e64 [[CC4:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC4]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[LOW_RESULT_REG:[0-9]+]], [[CONST]], v{{[0-9]+}}, [[CC1]]
; GCN: buffer_store_dwordx4 {{v\[}}[[LOW_RESULT_REG]]:
define amdgpu_kernel void @dynamic_insertelement_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x float> %a, float 5.000000e+00, i32 %b
  store <4 x float> %vecins, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v8f32:
; GCN-DAG: v_mov_b32_e32 [[CONST:v[0-9]+]], 0x40a00000
; GCN-DAG: v_cmp_ne_u32_e64 [[CCL:[^,]+]], [[IDX:s[0-9]+]], 7
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CCL]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, [[CONST]], v{{[0-9]+}}, [[CC1]]
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x float> %a, float 5.000000e+00, i32 %b
  store <8 x float> %vecins, <8 x float> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v16f32:
; GCN: v_movreld_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v16f32(<16 x float> addrspace(1)* %out, <16 x float> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x float> %a, float 5.000000e+00, i32 %b
  store <16 x float> %vecins, <16 x float> addrspace(1)* %out, align 64
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2i32:
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[LOW_RESULT_REG:[0-9]+]], 5, v{{[0-9]+}}, [[CC1]]
; GCN: buffer_store_dwordx2 {{v\[}}[[LOW_RESULT_REG]]:
define amdgpu_kernel void @dynamic_insertelement_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i32> %a, i32 5, i32 %b
  store <2 x i32> %vecins, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v3i32:
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX:s[0-9]+]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: buffer_store_dwordx3 v
define amdgpu_kernel void @dynamic_insertelement_v3i32(<3 x i32> addrspace(1)* %out, <3 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <3 x i32> %a, i32 5, i32 %b
  store <3 x i32> %vecins, <3 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v4i32:
; GCN: s_load_dword [[SVAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0x11|0x44}}
; GCN-DAG: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[SVAL]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC4:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[VVAL]], [[CC4]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC3:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}},  v{{[0-9]+}}, [[VVAL]], [[CC3]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[VVAL]], [[CC2]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[VVAL]], [[CC1]]
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> %a, i32 %b, [8 x i32], i32 %val) nounwind {
  %vecins = insertelement <4 x i32> %a, i32 %val, i32 %b
  store <4 x i32> %vecins, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v8i32:
; GCN-DAG: v_cmp_ne_u32_e64 [[CCL:[^,]+]], [[IDX:s[0-9]+]], 7
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CCL]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC1]]
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v8i32(<8 x i32> addrspace(1)* %out, <8 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <8 x i32> %a, i32 5, i32 %b
  store <8 x i32> %vecins, <8 x i32> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v16i32:
; GCN: v_movreld_b32
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v16i32(<16 x i32> addrspace(1)* %out, <16 x i32> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x i32> %a, i32 5, i32 %b
  store <16 x i32> %vecins, <16 x i32> addrspace(1)* %out, align 64
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2i16:
define amdgpu_kernel void @dynamic_insertelement_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i16> %a, i16 5, i32 %b
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v3i16:
define amdgpu_kernel void @dynamic_insertelement_v3i16(<3 x i16> addrspace(1)* %out, <3 x i16> %a, i32 %b) nounwind {
  %vecins = insertelement <3 x i16> %a, i16 5, i32 %b
  store <3 x i16> %vecins, <3 x i16> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2i8:
; VI: s_load_dword [[LOAD:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x28
; VI-NEXT: s_load_dword [[IDX:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; VI-NOT: _load
; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: v_lshlrev_b16_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], -1
; VI: v_and_b32_e32 [[INSERT:v[0-9]+]], 5, [[MASK]]
; VI: v_xor_b32_e32 [[NOT_MASK:v[0-9]+]], -1, [[MASK]]
; VI: v_and_b32_e32 [[AND_NOT_MASK:v[0-9]+]], [[LOAD]], [[NOT_MASK]]
; VI: v_or_b32_e32 [[OR:v[0-9]+]], [[INSERT]], [[AND_NOT_MASK]]
; VI: buffer_store_short [[OR]]
define amdgpu_kernel void @dynamic_insertelement_v2i8(<2 x i8> addrspace(1)* %out, [8 x i32], <2 x i8> %a, [8 x i32], i32 %b) nounwind {
  %vecins = insertelement <2 x i8> %a, i8 5, i32 %b
  store <2 x i8> %vecins, <2 x i8> addrspace(1)* %out, align 8
  ret void
}

; FIXME: post legalize i16 and i32 shifts aren't merged because of
; isTypeDesirableForOp in SimplifyDemandedBits

; GCN-LABEL: {{^}}dynamic_insertelement_v3i8:
; VI: s_load_dword [[LOAD:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x28
; VI-NEXT: s_load_dword [[IDX:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; VI-NOT: _load

; VI: v_mov_b32_e32 [[V_LOAD:v[0-9]+]], [[LOAD]]
; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshl_b32 [[SHIFTED_MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; VI: s_andn2_b32 [[AND_NOT_MASK:s[0-9]+]], [[LOAD]],  [[SHIFTED_MASK]]
; VI: v_bfi_b32 [[BFI:v[0-9]+]], [[SHIFTED_MASK]], 5, [[V_LOAD]]
; VI: s_lshr_b32 [[HI2:s[0-9]+]], [[AND_NOT_MASK]], 16

; VI-DAG: buffer_store_short [[BFI]]
; VI-DAG: v_mov_b32_e32 [[V_HI2:v[0-9]+]], [[HI2]]
; VI: buffer_store_byte [[V_HI2]]
define amdgpu_kernel void @dynamic_insertelement_v3i8(<3 x i8> addrspace(1)* %out, [8 x i32], <3 x i8> %a, [8 x i32], i32 %b) nounwind {
  %vecins = insertelement <3 x i8> %a, i8 5, i32 %b
  store <3 x i8> %vecins, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v4i8:
; VI: s_load_dword [[LOAD:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x28
; VI-NEXT: s_load_dword [[IDX:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; VI-NOT: _load

; VI: v_mov_b32_e32 [[V_LOAD:v[0-9]+]], [[LOAD]]
; VI-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshl_b32 [[SHIFTED_MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; VI: v_bfi_b32 [[BFI:v[0-9]+]], [[SHIFTED_MASK]], 5, [[V_LOAD]]
; VI: buffer_store_dword [[BFI]]
define amdgpu_kernel void @dynamic_insertelement_v4i8(<4 x i8> addrspace(1)* %out, [8 x i32], <4 x i8> %a, [8 x i32], i32 %b) nounwind {
  %vecins = insertelement <4 x i8> %a, i8 5, i32 %b
  store <4 x i8> %vecins, <4 x i8> addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}s_dynamic_insertelement_v8i8:
; VI-NOT: {{buffer|flat|global}}_load
; VI-DAG: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; VI-DAG: s_load_dword [[IDX:s[0-9]]], s[4:5], 0x10
; VI-DAG: s_mov_b32 s[[MASK_HI:[0-9]+]], 0{{$}}
; VI-DAG: s_load_dwordx2 [[VEC:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; VI-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI-DAG: s_mov_b32 s[[MASK_LO:[0-9]+]], 0xffff
; VI: s_lshl_b64 s{{\[}}[[MASK_SHIFT_LO:[0-9]+]]:[[MASK_SHIFT_HI:[0-9]+]]{{\]}}, s{{\[}}[[MASK_LO]]:[[MASK_HI]]{{\]}}, [[SCALED_IDX]]
; VI: s_andn2_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[VEC]], s{{\[}}[[MASK_SHIFT_LO]]:[[MASK_SHIFT_HI]]{{\]}}
; VI: s_and_b32 s[[INS:[0-9]+]], s[[MASK_SHIFT_LO]], 5
; VI: s_or_b64 s{{\[}}[[RESULT0:[0-9]+]]:[[RESULT1:[0-9]+]]{{\]}}, s{{\[}}[[INS]]:[[MASK_HI]]{{\]}}, [[AND]]
; VI: v_mov_b32_e32 v[[V_RESULT0:[0-9]+]], s[[RESULT0]]
; VI: v_mov_b32_e32 v[[V_RESULT1:[0-9]+]], s[[RESULT1]]
; VI: buffer_store_dwordx2 v{{\[}}[[V_RESULT0]]:[[V_RESULT1]]{{\]}}
define amdgpu_kernel void @s_dynamic_insertelement_v8i8(<8 x i8> addrspace(1)* %out, <8 x i8> addrspace(4)* %a.ptr, i32 %b) nounwind {
  %a = load <8 x i8>, <8 x i8> addrspace(4)* %a.ptr, align 4
  %vecins = insertelement <8 x i8> %a, i8 5, i32 %b
  store <8 x i8> %vecins, <8 x i8> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v16i8:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx4
; GCN: s_load_dword s

; GCN-NOT: buffer_store_byte

; GCN-DAG: v_cmp_ne_u32_e64 [[CCL:[^,]+]], [[IDX:s[0-9]+]], 15
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CCL]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 5, v{{[0-9]+}}, [[CC1]]

; GCN: buffer_store_dwordx4
define amdgpu_kernel void @dynamic_insertelement_v16i8(<16 x i8> addrspace(1)* %out, <16 x i8> %a, i32 %b) nounwind {
  %vecins = insertelement <16 x i8> %a, i8 5, i32 %b
  store <16 x i8> %vecins, <16 x i8> addrspace(1)* %out, align 16
  ret void
}

; This test requires handling INSERT_SUBREG in SIFixSGPRCopies.  Check that
; the compiler doesn't crash.
; GCN-LABEL: {{^}}insert_split_bb:
define amdgpu_kernel void @insert_split_bb(<2 x i32> addrspace(1)* %out, i32 addrspace(1)* %in, i32 %a, i32 %b) {
entry:
  %0 = insertelement <2 x i32> undef, i32 %a, i32 0
  %1 = icmp eq i32 %a, 0
  br i1 %1, label %if, label %else

if:
  %2 = load i32, i32 addrspace(1)* %in
  %3 = insertelement <2 x i32> %0, i32 %2, i32 1
  br label %endif

else:
  %4 = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %5 = load i32, i32 addrspace(1)* %4
  %6 = insertelement <2 x i32> %0, i32 %5, i32 1
  br label %endif

endif:
  %7 = phi <2 x i32> [%3, %if], [%6, %else]
  store <2 x i32> %7, <2 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2f64:
; GCN-DAG: s_load_dwordx4 s{{\[}}[[A_ELT0:[0-9]+]]:[[A_ELT3:[0-9]+]]{{\]}}
; GCN-DAG: s_load_dword [[IDX:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, {{0x18|0x60}}{{$}}

; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 [[ELT1:v[0-9]+]], 0x40200000

; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ELT1]], [[CC2]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC2]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[ELT1]], [[CC1]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC1]]

; GCN: buffer_store_dwordx4
; GCN: s_endpgm
define amdgpu_kernel void @dynamic_insertelement_v2f64(<2 x double> addrspace(1)* %out, [8 x i32], <2 x double> %a, [8 x i32], i32 %b) nounwind {
  %vecins = insertelement <2 x double> %a, double 8.0, i32 %b
  store <2 x double> %vecins, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v2i64:

; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 5, [[CC2]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC2]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 5, [[CC1]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC1]]

; GCN: buffer_store_dwordx4
; GCN: s_endpgm
define amdgpu_kernel void @dynamic_insertelement_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> %a, i32 %b) nounwind {
  %vecins = insertelement <2 x i64> %a, i64 5, i32 %b
  store <2 x i64> %vecins, <2 x i64> addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v3i64:
; GCN-DAG: v_cmp_eq_u32_e64 [[CC3:[^,]+]], [[IDX:s[0-9]+]], 2
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}},  v{{[0-9]+}}, 5, [[CC3]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}},  v{{[0-9]+}}, 0, [[CC3]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 5, [[CC2]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC2]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 5, [[CC1]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC1]]
define amdgpu_kernel void @dynamic_insertelement_v3i64(<3 x i64> addrspace(1)* %out, <3 x i64> %a, i32 %b) nounwind {
  %vecins = insertelement <3 x i64> %a, i64 5, i32 %b
  store <3 x i64> %vecins, <3 x i64> addrspace(1)* %out, align 32
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v4f64:

; GCN-DAG: v_mov_b32_e32 [[CONST:v[0-9]+]], 0x40200000
; GCN-DAG: v_cmp_eq_u32_e64 [[CC4:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[CONST]], [[CC4]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC4]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC3:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}},  v{{[0-9]+}}, [[CONST]], [[CC3]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}},  v{{[0-9]+}}, 0, [[CC3]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[CONST]], [[CC2]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC2]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[CONST]], [[CC1]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC1]]

; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
; GCN: ScratchSize: 0

define amdgpu_kernel void @dynamic_insertelement_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, i32 %b) nounwind {
  %vecins = insertelement <4 x double> %a, double 8.0, i32 %b
  store <4 x double> %vecins, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}dynamic_insertelement_v8f64:
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:64{{$}}
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:80{{$}}
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:96{{$}}
; GCN-DAG: buffer_store_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:112{{$}}

; GCN: buffer_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, s[0:3], {{s[0-9]+}} offen{{$}}

; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:64{{$}}
; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:80{{$}}
; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:96{{$}}
; GCN-DAG: buffer_load_dwordx4 v{{\[[0-9]+:[0-9]+\]}}, off, s[0:3], {{s[0-9]+}} offset:112{{$}}

; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
; GCN: ScratchSize: 128
define amdgpu_kernel void @dynamic_insertelement_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, i32 %b) #0 {
  %vecins = insertelement <8 x double> %a, double 8.0, i32 %b
  store <8 x double> %vecins, <8 x double> addrspace(1)* %out, align 16
  ret void
}

declare <4 x float> @llvm.amdgcn.image.gather4.lz.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
