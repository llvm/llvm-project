; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s

declare double @llvm.fabs.f64(double) #0
declare double @llvm.floor.f64(double) #0

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
define void @fract_f64(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %floor.x = call double @llvm.floor.f64(double %x)
  %fract = fsub double %x, %floor.x
  store double %fract, double addrspace(1)* %out
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
define void @fract_f64_neg(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %neg.x = fsub double -0.0, %x
  %floor.neg.x = call double @llvm.floor.f64(double %neg.x)
  %fract = fsub double %neg.x, %floor.neg.x
  store double %fract, double addrspace(1)* %out
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
define void @fract_f64_neg_abs(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %abs.x = call double @llvm.fabs.f64(double %x)
  %neg.abs.x = fsub double -0.0, %abs.x
  %floor.neg.abs.x = call double @llvm.floor.f64(double %neg.abs.x)
  %fract = fsub double %neg.abs.x, %floor.neg.abs.x
  store double %fract, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
