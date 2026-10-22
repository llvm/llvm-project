; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck --check-prefixes=SI,FUNC %s
; RUN: llc -mtriple=amdgpu7.04 < %s | FileCheck --check-prefixes=CI,FUNC %s
; RUN: llc -mtriple=amdgpu8.02 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=CI,FUNC %s

declare double @llvm.fabs.f64(double) #0
declare double @llvm.floor.f64(double) #0

; FUNC-LABEL: {{^}}fract_f64:
; SI-DAG: v_fract_f64_e32 [[FRC:v\[[0-9]+:[0-9]+\]]], v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v[[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], [[FRC]], v[[[UPLO]]:[[UPHI]]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v[[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], v[[[LO]]:[[HI]]], -v[[[RESLO]]:[[RESHI]]]
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], v[[[LO]]:[[HI]]], -[[SUB0]]

; CI: v_floor_f64_e32 [[FLOORX:v\[[0-9]+:[0-9]+\]]], v[0:1]
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], v[0:1], -[[FLOORX]]

define double @fract_f64(double %src) #1 {
  %floor.x = call double @llvm.floor.f64(double %src)
  %fract = fsub double %src, %floor.x
  ret double %fract
}

; FUNC-LABEL: {{^}}fract_f64_neg:
; SI-DAG: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v[[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], [[FRC]], v[[[UPLO]]:[[UPHI]]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v[[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], -v[[[LO]]:[[HI]]], -v[[[RESLO]]:[[RESHI]]]
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -v[[[LO]]:[[HI]]], -[[SUB0]]

; CI: v_floor_f64_e64 [[FLOORX:v\[[0-9]+:[0-9]+\]]], -v[0:1]
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -v[0:1], -[[FLOORX]]

define double @fract_f64_neg(double %src) #1 {
  %neg.x = fneg double %src
  %floor.neg.x = call double @llvm.floor.f64(double %neg.x)
  %fract = fsub double %neg.x, %floor.neg.x
  ret double %fract
}

; FUNC-LABEL: {{^}}fract_f64_neg_abs:
; SI-DAG: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -|v[[[LO:[0-9]+]]:[[HI:[0-9]+]]]|
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v[[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], [[FRC]], v[[[UPLO]]:[[UPHI]]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v[[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], -|v[[[LO]]:[[HI]]]|, -v[[[RESLO]]:[[RESHI]]]
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -|v[[[LO]]:[[HI]]]|, -[[SUB0]]

; CI: v_floor_f64_e64 [[FLOORX:v\[[0-9]+:[0-9]+\]]], -|v[0:1]|
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -|v[0:1]|, -[[FLOORX]]

define double @fract_f64_neg_abs(double %src) #1 {
  %abs.x = call double @llvm.fabs.f64(double %src)
  %neg.abs.x = fneg double %abs.x
  %floor.neg.abs.x = call double @llvm.floor.f64(double %neg.abs.x)
  %fract = fsub double %neg.abs.x, %floor.neg.abs.x
  ret double %fract
}

; FUNC-LABEL: {{^}}multi_use_floor_fract_f64:
define amdgpu_kernel void @multi_use_floor_fract_f64(ptr addrspace(1) %out, ptr addrspace(1) %src) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %src.tid = getelementptr inbounds double, ptr addrspace(1) %src, i32 %tid
  %x = load double, ptr addrspace(1) %src.tid
  %floor.x = call double @llvm.floor.f64(double %x)
  %fract = fsub double %x, %floor.x
  store volatile double %floor.x, ptr addrspace(1) %out
  store volatile double %fract, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
