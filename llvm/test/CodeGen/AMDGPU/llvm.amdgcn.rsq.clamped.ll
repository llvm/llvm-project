; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s

declare double @llvm.amdgcn.rsq.clamped.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}rsq_clamped_f64:
; SI: v_rsq_clamp_f64_e32

; VI: v_rsq_f64_e32 [[RSQ:v\[[0-9]+:[0-9]+\]]], s[2:3]
; TODO: this constant should be folded:
; VI: s_mov_b32 s[[ALLBITS:[0-9+]]], -1
; VI: s_mov_b32 s[[HIGH1:[0-9+]]], 0x7fefffff
; VI: s_mov_b32 s[[LOW1:[0-9+]]], s[[ALLBITS]]
; VI: v_min_f64 v[0:1], [[RSQ]], s{{\[}}[[LOW1]]:[[HIGH1]]]
; VI: s_mov_b32 s[[HIGH2:[0-9+]]], 0xffefffff
; VI: s_mov_b32 s[[LOW2:[0-9+]]], s[[ALLBITS]]
; VI: v_max_f64 v[0:1], v[0:1], s{{\[}}[[LOW2]]:[[HIGH2]]]

define void @rsq_clamped_f64(double addrspace(1)* %out, double %src) nounwind {
  %rsq_clamped = call double @llvm.amdgcn.rsq.clamped.f64(double %src) nounwind readnone
  store double %rsq_clamped, double addrspace(1)* %out, align 8
  ret void
}
