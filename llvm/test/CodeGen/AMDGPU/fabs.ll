; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s


; DAGCombiner will transform:
; (fabs (f32 bitcast (i32 a))) => (f32 bitcast (and (i32 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; FUNC-LABEL: {{^}}fabs_fn_free:
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|

; GCN: v_and_b32

define amdgpu_kernel void @fabs_fn_free(float addrspace(1)* %out, i32 %in) {
  %bc= bitcast i32 %in to float
  %fabs = call float @fabs(float %bc)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_free:
; R600-NOT: AND
; R600: |PV.{{[XYZW]}}|

; GCN: v_and_b32

define amdgpu_kernel void @fabs_free(float addrspace(1)* %out, i32 %in) {
  %bc= bitcast i32 %in to float
  %fabs = call float @llvm.fabs.f32(float %bc)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; GCN: v_and_b32
define amdgpu_kernel void @fabs_f32(float addrspace(1)* %out, float %in) {
  %fabs = call float @llvm.fabs.f32(float %in)
  store float %fabs, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v2f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; GCN: v_and_b32
; GCN: v_and_b32
define amdgpu_kernel void @fabs_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
  %fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %in)
  store <2 x float> %fabs, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_v4f32:
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|
; R600: |{{(PV|T[0-9])\.[XYZW]}}|

; GCN: v_and_b32
; GCN: v_and_b32
; GCN: v_and_b32
; GCN: v_and_b32
define amdgpu_kernel void @fabs_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
  %fabs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %in)
  store <4 x float> %fabs, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fabs_fn_fold:
; SI: s_load_dword [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN-NOT: and
; GCN: v_mul_f32_e64 v{{[0-9]+}}, |[[ABS_VALUE]]|, v{{[0-9]+}}
define amdgpu_kernel void @fabs_fn_fold(float addrspace(1)* %out, float %in0, float %in1) {
  %fabs = call float @fabs(float %in0)
  %fmul = fmul float %fabs, %in1
  store float %fmul, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fabs_fold:
; SI: s_load_dword [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0xb
; VI: s_load_dword [[ABS_VALUE:s[0-9]+]], s[{{[0-9]+:[0-9]+}}], 0x2c
; GCN-NOT: and
; GCN: v_mul_f32_e64 v{{[0-9]+}}, |[[ABS_VALUE]]|, v{{[0-9]+}}
define amdgpu_kernel void @fabs_fold(float addrspace(1)* %out, float %in0, float %in1) {
  %fabs = call float @llvm.fabs.f32(float %in0)
  %fmul = fmul float %fabs, %in1
  store float %fmul, float addrspace(1)* %out
  ret void
}

; Make sure we turn some integer operations back into fabs
; FUNC-LABEL: {{^}}bitpreserve_fabs_f32:
; GCN: v_add_f32_e64 v{{[0-9]+}}, |s{{[0-9]+}}|, 1.0
define amdgpu_kernel void @bitpreserve_fabs_f32(float addrspace(1)* %out, float %in) {
  %in.bc = bitcast float %in to i32
  %int.abs = and i32 %in.bc, 2147483647
  %bc = bitcast i32 %int.abs to float
  %fadd = fadd float %bc, 1.0
  store float %fadd, float addrspace(1)* %out
  ret void
}

declare float @fabs(float) readnone
declare float @llvm.fabs.f32(float) readnone
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float>) readnone
