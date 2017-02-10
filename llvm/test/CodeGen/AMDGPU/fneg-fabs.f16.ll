; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s

; GCN-LABEL: {{^}}fneg_fabs_fadd_f16:
; CI: v_cvt_f32_f16_e32
; CI: v_cvt_f32_f16_e64 [[CVT_ABS_X:v[0-9]+]], |v{{[0-9]+}}|
; CI: v_subrev_f32_e32 v{{[0-9]+}}, [[CVT_ABS_X]], v{{[0-9]+}}

; VI-NOT: _and
; VI: v_sub_f16_e64 {{v[0-9]+}}, {{v[0-9]+}}, |{{v[0-9]+}}|
define void @fneg_fabs_fadd_f16(half addrspace(1)* %out, half %x, half %y) {
  %fabs = call half @llvm.fabs.f16(half %x)
  %fsub = fsub half -0.000000e+00, %fabs
  %fadd = fadd half %y, %fsub
  store half %fadd, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_fmul_f16:
; CI-DAG: v_cvt_f32_f16_e32
; CI-DAG: v_cvt_f32_f16_e64 [[CVT_NEG_ABS_X:v[0-9]+]], -|{{v[0-9]+}}|
; CI: v_mul_f32_e32 {{v[0-9]+}}, [[CVT_NEG_ABS_X]], {{v[0-9]+}}
; CI: v_cvt_f16_f32_e32

; VI-NOT: _and
; VI: v_mul_f16_e64 [[MUL:v[0-9]+]], {{v[0-9]+}}, -|{{v[0-9]+}}|
; VI-NOT: [[MUL]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[MUL]]
define void @fneg_fabs_fmul_f16(half addrspace(1)* %out, half %x, half %y) {
  %fabs = call half @llvm.fabs.f16(half %x)
  %fsub = fsub half -0.000000e+00, %fabs
  %fmul = fmul half %y, %fsub
  store half %fmul, half addrspace(1)* %out, align 2
  ret void
}

; DAGCombiner will transform:
; (fabs (f16 bitcast (i16 a))) => (f16 bitcast (and (i16 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; GCN-LABEL: {{^}}fneg_fabs_free_f16:
; GCN: v_or_b32_e32 v{{[0-9]+}}, 0x8000, v{{[0-9]+}}
define void @fneg_fabs_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc = bitcast i16 %in to half
  %fabs = call half @llvm.fabs.f16(half %bc)
  %fsub = fsub half -0.000000e+00, %fabs
  store half %fsub, half addrspace(1)* %out
  ret void
}

; FIXME: Should use or
; GCN-LABEL: {{^}}fneg_fabs_f16:
; GCN: v_or_b32_e32 v{{[0-9]+}}, 0x8000, v{{[0-9]+}}
define void @fneg_fabs_f16(half addrspace(1)* %out, half %in) {
  %fabs = call half @llvm.fabs.f16(half %in)
  %fsub = fsub half -0.000000e+00, %fabs
  store half %fsub, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}v_fneg_fabs_f16:
; GCN: v_or_b32_e32 v{{[0-9]+}}, 0x8000, v{{[0-9]+}}
define void @v_fneg_fabs_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %val = load half, half addrspace(1)* %in, align 2
  %fabs = call half @llvm.fabs.f16(half %val)
  %fsub = fsub half -0.000000e+00, %fabs
  store half %fsub, half addrspace(1)* %out, align 2
  ret void
}

; FIXME: single bit op
; GCN-LABEL: {{^}}fneg_fabs_v2f16:
; GCN: s_mov_b32 [[MASK:s[0-9]+]], 0x8000{{$}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: store_dword
define void @fneg_fabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in) {
  %fabs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %in)
  %fsub = fsub <2 x half> <half -0.000000e+00, half -0.000000e+00>, %fabs
  store <2 x half> %fsub, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fneg_fabs_v4f16:
; GCN: s_mov_b32 [[MASK:s[0-9]+]], 0x8000{{$}}
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: v_or_b32_e32 v{{[0-9]+}}, [[MASK]],
; GCN: store_dwordx2
define void @fneg_fabs_v4f16(<4 x half> addrspace(1)* %out, <4 x half> %in) {
  %fabs = call <4 x half> @llvm.fabs.v4f16(<4 x half> %in)
  %fsub = fsub <4 x half> <half -0.000000e+00, half -0.000000e+00, half -0.000000e+00, half -0.000000e+00>, %fabs
  store <4 x half> %fsub, <4 x half> addrspace(1)* %out
  ret void
}

declare half @llvm.fabs.f16(half) readnone
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) readnone
declare <4 x half> @llvm.fabs.v4f16(<4 x half>) readnone
