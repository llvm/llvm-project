; RUN: llc -march=amdgcn -mcpu=kaveri -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mtriple=amdgcn--amdhsa -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s

; FIXME: Should be able to do scalar op
; FUNC-LABEL: {{^}}s_fneg_f16:

define void @s_fneg_f16(half addrspace(1)* %out, half %in) {
  %fneg = fsub half -0.000000e+00, %in
  store half %fneg, half addrspace(1)* %out
  ret void
}

; FIXME: Should be able to use bit operations when illegal type as
; well.

; FUNC-LABEL: {{^}}v_fneg_f16:
; GCN: flat_load_ushort [[VAL:v[0-9]+]],
; GCN: v_xor_b32_e32 [[XOR:v[0-9]+]], 0x8000, [[VAL]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[XOR]]
; SI: buffer_store_short [[XOR]]
define void @v_fneg_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %val = load half, half addrspace(1)* %in, align 2
  %fneg = fsub half -0.000000e+00, %val
  store half %fneg, half addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fneg_free_f16:
; GCN: flat_load_ushort [[NEG_VALUE:v[0-9]+]],

; XCI: s_xor_b32 [[XOR:s[0-9]+]], [[NEG_VALUE]], 0x8000{{$}}
; CI: v_xor_b32_e32 [[XOR:v[0-9]+]], 0x8000, [[NEG_VALUE]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[XOR]]
define void @fneg_free_f16(half addrspace(1)* %out, i16 %in) {
  %bc = bitcast i16 %in to half
  %fsub = fsub half -0.0, %bc
  store half %fsub, half addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_fneg_fold_f16:
; GCN: flat_load_ushort [[NEG_VALUE:v[0-9]+]]

; CI-DAG: v_cvt_f32_f16_e32 [[CVT_VAL:v[0-9]+]], [[NEG_VALUE]]
; CI-DAG: v_cvt_f32_f16_e64 [[NEG_CVT0:v[0-9]+]], -[[NEG_VALUE]]
; CI: v_mul_f32_e32 [[MUL:v[0-9]+]], [[CVT_VAL]], [[NEG_CVT0]]
; CI: v_cvt_f16_f32_e32 [[CVT1:v[0-9]+]], [[MUL]]
; CI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[CVT1]]

; VI-NOT: [[NEG_VALUE]]
; VI: v_mul_f16_e64 v{{[0-9]+}}, -[[NEG_VALUE]], [[NEG_VALUE]]
define void @v_fneg_fold_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %val = load half, half addrspace(1)* %in
  %fsub = fsub half -0.0, %val
  %fmul = fmul half %fsub, %val
  store half %fmul, half addrspace(1)* %out
  ret void
}
