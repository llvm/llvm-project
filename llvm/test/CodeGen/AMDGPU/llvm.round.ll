; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck --check-prefixes=GCN,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN,GFX89,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCN,GFX89,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=R600,FUNC %s

; FUNC-LABEL: {{^}}round_f32:
; GCN-DAG: s_load_dword [[SX:s[0-9]+]]
; GCN-DAG: s_brev_b32 [[K:s[0-9]+]], -2{{$}}
; GCN-DAG: v_trunc_f32_e32 [[TRUNC:v[0-9]+]], [[SX]]
; GCN-DAG: v_sub_f32_e32 [[SUB:v[0-9]+]], [[SX]], [[TRUNC]]
; GCN-DAG: v_mov_b32_e32 [[VX:v[0-9]+]], [[SX]]
; GCN: v_bfi_b32 [[COPYSIGN:v[0-9]+]], [[K]], 1.0, [[VX]]
; GCN: v_cmp_ge_f32_e64 vcc, |[[SUB]]|, 0.5
; GCN: v_cndmask_b32_e32 [[SEL:v[0-9]+]], 0, [[VX]]
; GCN: v_add_f32_e32 [[RESULT:v[0-9]+]], [[TRUNC]], [[SEL]]
; GCN: buffer_store_dword [[RESULT]]

; R600: TRUNC {{.*}}, [[ARG:KC[0-9]\[[0-9]+\]\.[XYZW]]]
; R600-DAG: ADD  {{.*}},
; R600-DAG: BFI_INT
; R600-DAG: SETGE
; R600-DAG: CNDE
; R600-DAG: ADD
define amdgpu_kernel void @round_f32(ptr addrspace(1) %out, float %x) #0 {
  %result = call float @llvm.round.f32(float %x) #1
  store float %result, ptr addrspace(1) %out
  ret void
}

; The vector tests are really difficult to verify, since it can be hard to
; predict how the scheduler will order the instructions.  We already have
; a test for the scalar case, so the vector tests just check that the
; compiler doesn't crash.

; FUNC-LABEL: {{^}}round_v2f32:
; GCN: s_endpgm
; R600: CF_END
define amdgpu_kernel void @round_v2f32(ptr addrspace(1) %out, <2 x float> %in) #0 {
  %result = call <2 x float> @llvm.round.v2f32(<2 x float> %in) #1
  store <2 x float> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}round_v4f32:
; GCN: s_endpgm
; R600: CF_END
define amdgpu_kernel void @round_v4f32(ptr addrspace(1) %out, <4 x float> %in) #0 {
  %result = call <4 x float> @llvm.round.v4f32(<4 x float> %in) #1
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}round_v8f32:
; GCN: s_endpgm
; R600: CF_END
define amdgpu_kernel void @round_v8f32(ptr addrspace(1) %out, <8 x float> %in) #0 {
  %result = call <8 x float> @llvm.round.v8f32(<8 x float> %in) #1
  store <8 x float> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}round_f16:
; GFX89-DAG: s_load_dword [[SX:s[0-9]+]]
; GFX89-DAG: s_movk_i32 [[K:s[0-9]+]], 0x7fff{{$}}
; GFX89-DAG: v_mov_b32_e32 [[VX:v[0-9]+]], [[SX]]
; GFX89-DAG: v_mov_b32_e32 [[BFI_K:v[0-9]+]], 0x3c00
; GFX89: v_bfi_b32 [[COPYSIGN:v[0-9]+]], [[K]], [[BFI_K]], [[VX]]

; GFX89: v_trunc_f16_e32 [[TRUNC:v[0-9]+]], [[SX]]
; GFX89: v_sub_f16_e32 [[SUB:v[0-9]+]], [[SX]], [[TRUNC]]
; GFX89: v_cmp_ge_f16_e64 vcc, |[[SUB]]|, 0.5
; GFX89: v_cndmask_b32_e32 [[SEL:v[0-9]+]], 0, [[COPYSIGN]]
; GFX89: v_add_f16_e32 [[RESULT:v[0-9]+]], [[TRUNC]], [[SEL]]
; GFX89: buffer_store_short [[RESULT]]
define amdgpu_kernel void @round_f16(ptr addrspace(1) %out, i32 %x.arg) #0 {
  %x.arg.trunc = trunc i32 %x.arg to i16
  %x = bitcast i16 %x.arg.trunc to half
  %result = call half @llvm.round.f16(half %x) #1
  store half %result, ptr addrspace(1) %out
  ret void
}

; Should be scalarized
; FUNC-LABEL: {{^}}round_v2f16:
; GFX89-DAG: s_movk_i32 [[K:s[0-9]+]], 0x7fff{{$}}
; GFX89-DAG: v_mov_b32_e32 [[BFI_K:v[0-9]+]], 0x3c00
; GFX89: v_bfi_b32 [[COPYSIGN0:v[0-9]+]], [[K]], [[BFI_K]],
; GFX89: v_bfi_b32 [[COPYSIGN1:v[0-9]+]], [[K]], [[BFI_K]],

; GFX9: v_pack_b32_f16
define amdgpu_kernel void @round_v2f16(ptr addrspace(1) %out, i32 %in.arg) #0 {
  %in = bitcast i32 %in.arg to <2 x half>
  %result = call <2 x half> @llvm.round.v2f16(<2 x half> %in)
  store <2 x half> %result, ptr addrspace(1) %out
  ret void
}

declare float @llvm.round.f32(float) #1
declare <2 x float> @llvm.round.v2f32(<2 x float>) #1
declare <4 x float> @llvm.round.v4f32(<4 x float>) #1
declare <8 x float> @llvm.round.v8f32(<8 x float>) #1

declare half @llvm.round.f16(half) #1
declare <2 x half> @llvm.round.v2f16(<2 x half>) #1
declare <4 x half> @llvm.round.v4f16(<4 x half>) #1
declare <8 x half> @llvm.round.v8f16(<8 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
