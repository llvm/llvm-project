; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}round_f32:
; SI-DAG: s_load_dword [[SX:s[0-9]+]]
; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x7fffffff
; SI-DAG: v_trunc_f32_e32 [[TRUNC:v[0-9]+]], [[SX]]
; SI-DAG: v_sub_f32_e32 [[SUB:v[0-9]+]], [[SX]], [[TRUNC]]
; SI-DAG: v_mov_b32_e32 [[VX:v[0-9]+]], [[SX]]
; SI: v_bfi_b32 [[COPYSIGN:v[0-9]+]], [[K]], 1.0, [[VX]]
; SI: v_cmp_le_f32_e64 vcc, 0.5, |[[SUB]]|
; SI: v_cndmask_b32_e32 [[SEL:v[0-9]+]], 0, [[VX]]
; SI: v_add_f32_e32 [[RESULT:v[0-9]+]], [[SEL]], [[TRUNC]]
; SI: buffer_store_dword [[RESULT]]

; R600: TRUNC {{.*}}, [[ARG:KC[0-9]\[[0-9]+\]\.[XYZW]]]
; R600-DAG: ADD  {{.*}},
; R600-DAG: BFI_INT
; R600-DAG: SETGE
; R600-DAG: CNDE
; R600-DAG: ADD
define void @round_f32(float addrspace(1)* %out, float %x) #0 {
  %result = call float @llvm.round.f32(float %x) #1
  store float %result, float addrspace(1)* %out
  ret void
}

; The vector tests are really difficult to verify, since it can be hard to
; predict how the scheduler will order the instructions.  We already have
; a test for the scalar case, so the vector tests just check that the
; compiler doesn't crash.

; FUNC-LABEL: {{^}}round_v2f32:
; SI: s_endpgm
; R600: CF_END
define void @round_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) #0 {
  %result = call <2 x float> @llvm.round.v2f32(<2 x float> %in) #1
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}round_v4f32:
; SI: s_endpgm
; R600: CF_END
define void @round_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) #0 {
  %result = call <4 x float> @llvm.round.v4f32(<4 x float> %in) #1
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}round_v8f32:
; SI: s_endpgm
; R600: CF_END
define void @round_v8f32(<8 x float> addrspace(1)* %out, <8 x float> %in) #0 {
  %result = call <8 x float> @llvm.round.v8f32(<8 x float> %in) #1
  store <8 x float> %result, <8 x float> addrspace(1)* %out
  ret void
}

declare float @llvm.round.f32(float) #1
declare <2 x float> @llvm.round.v2f32(<2 x float>) #1
declare <4 x float> @llvm.round.v4f32(<4 x float>) #1
declare <8 x float> @llvm.round.v8f32(<8 x float>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
