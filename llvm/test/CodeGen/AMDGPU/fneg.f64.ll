; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fneg_f64:
; GCN: v_xor_b32
define void @fneg_f64(double addrspace(1)* %out, double %in) {
  %fneg = fsub double -0.000000e+00, %in
  store double %fneg, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fneg_v2f64:
; GCN: v_xor_b32
; GCN: v_xor_b32
define void @fneg_v2f64(<2 x double> addrspace(1)* nocapture %out, <2 x double> %in) {
  %fneg = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %in
  store <2 x double> %fneg, <2 x double> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fneg_v4f64:
; R600: -PV
; R600: -T
; R600: -PV
; R600: -PV

; GCN: v_xor_b32
; GCN: v_xor_b32
; GCN: v_xor_b32
; GCN: v_xor_b32
define void @fneg_v4f64(<4 x double> addrspace(1)* nocapture %out, <4 x double> %in) {
  %fneg = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %in
  store <4 x double> %fneg, <4 x double> addrspace(1)* %out
  ret void
}

; DAGCombiner will transform:
; (fneg (f64 bitcast (i64 a))) => (f64 bitcast (xor (i64 a), 0x80000000))
; unless the target returns true for isNegFree()

; FUNC-LABEL: {{^}}fneg_free_f64:
; GCN: v_add_f64 {{v\[[0-9]+:[0-9]+\]}}, -{{s\[[0-9]+:[0-9]+\]}}, 0{{$}}
define void @fneg_free_f64(double addrspace(1)* %out, i64 %in) {
  %bc = bitcast i64 %in to double
  %fsub = fsub double 0.0, %bc
  store double %fsub, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fneg_fold_f64:
; SI: s_load_dwordx2 [[NEG_VALUE:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[NEG_VALUE:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-NOT: xor
; GCN: v_mul_f64 {{v\[[0-9]+:[0-9]+\]}}, -[[NEG_VALUE]], [[NEG_VALUE]]
define void @fneg_fold_f64(double addrspace(1)* %out, double %in) {
  %fsub = fsub double -0.0, %in
  %fmul = fmul double %fsub, %in
  store double %fmul, double addrspace(1)* %out
  ret void
}
