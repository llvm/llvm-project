; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s

declare double @llvm.AMDGPU.rsq.clamped.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}rsq_clamped_f64:
; SI: v_rsq_clamp_f64_e32

; VI-DAG: v_rsq_f64_e32 [[RSQ:v\[[0-9]+:[0-9]+\]]], s[{{[0-9]+:[0-9]+}}]
; TODO: this constant should be folded:
; VI-DAG: s_mov_b32 s[[LOW1:[0-9+]]], -1
; VI-DAG: s_mov_b32 s[[HIGH1:[0-9+]]], 0x7fefffff
; VI-DAG: v_min_f64 v[0:1], [[RSQ]], s{{\[}}[[LOW1]]:[[HIGH1]]]
; VI-DAG: s_mov_b32 s[[HIGH2:[0-9+]]], 0xffefffff
; VI-DAG: v_max_f64 v[0:1], v[0:1], s{{\[}}[[LOW1]]:[[HIGH2]]]

define void @rsq_clamped_f64(double addrspace(1)* %out, double %src) nounwind {
  %rsq_clamped = call double @llvm.AMDGPU.rsq.clamped.f64(double %src) nounwind readnone
  store double %rsq_clamped, double addrspace(1)* %out, align 8
  ret void
}
