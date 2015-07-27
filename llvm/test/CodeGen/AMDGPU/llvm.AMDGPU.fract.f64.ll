; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s

declare double @llvm.fabs.f64(double %Val)
declare double @llvm.AMDGPU.fract.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}fract_f64:
; GCN: v_fract_f64_e32 [[FRC:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI: v_cmp_class_f64_e64 [[COND:s\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e64 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], [[COND]]
; SI: v_cndmask_b32_e64 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], [[COND]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESLO]]:[[RESHI]]]
; CI: buffer_store_dwordx2 [[FRC]]
define void @fract_f64(double addrspace(1)* %out, double addrspace(1)* %src) nounwind {
  %val = load double, double addrspace(1)* %src, align 4
  %fract = call double @llvm.AMDGPU.fract.f64(double %val) nounwind readnone
  store double %fract, double addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fract_f64_neg:
; GCN: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI: v_cmp_class_f64_e64 [[COND:s\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e64 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], [[COND]]
; SI: v_cndmask_b32_e64 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], [[COND]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESLO]]:[[RESHI]]]
; CI: buffer_store_dwordx2 [[FRC]]
define void @fract_f64_neg(double addrspace(1)* %out, double addrspace(1)* %src) nounwind {
  %val = load double, double addrspace(1)* %src, align 4
  %neg = fsub double 0.0, %val
  %fract = call double @llvm.AMDGPU.fract.f64(double %neg) nounwind readnone
  store double %fract, double addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}fract_f64_neg_abs:
; GCN: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -|v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]|
; SI: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI: v_cmp_class_f64_e64 [[COND:s\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e64 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], [[COND]]
; SI: v_cndmask_b32_e64 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], [[COND]]
; SI: buffer_store_dwordx2 v{{\[}}[[RESLO]]:[[RESHI]]]
; CI: buffer_store_dwordx2 [[FRC]]
define void @fract_f64_neg_abs(double addrspace(1)* %out, double addrspace(1)* %src) nounwind {
  %val = load double, double addrspace(1)* %src, align 4
  %abs = call double @llvm.fabs.f64(double %val)
  %neg = fsub double 0.0, %abs
  %fract = call double @llvm.AMDGPU.fract.f64(double %neg) nounwind readnone
  store double %fract, double addrspace(1)* %out, align 4
  ret void
}
