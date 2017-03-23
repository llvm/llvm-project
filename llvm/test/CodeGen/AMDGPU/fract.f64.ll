; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=FUNC %s

; RUN: llc -march=amdgcn -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN-UNSAFE -check-prefix=SI-UNSAFE -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=GCN-UNSAFE -check-prefix=VI-UNSAFE -check-prefix=FUNC %s

declare double @llvm.fabs.f64(double) #0
declare double @llvm.floor.f64(double) #0

; FUNC-LABEL: {{^}}fract_f64:
; SI-DAG: v_fract_f64_e32 [[FRC:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO]]:[[HI]]{{\]}}, -v{{\[}}[[RESLO]]:[[RESHI]]{{\]}}
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], v{{\[}}[[LO]]:[[HI]]{{\]}}, -[[SUB0]]

; CI: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; CI: v_floor_f64_e32 [[FLOORX:v\[[0-9]+:[0-9]+\]]], [[X]]
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], [[X]], -[[FLOORX]]

; GCN-UNSAFE: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; GCN-UNSAFE: v_fract_f64_e32 [[FRACT:v\[[0-9]+:[0-9]+\]]], [[X]]

; GCN: buffer_store_dwordx2 [[FRACT]]
define amdgpu_kernel void @fract_f64(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %floor.x = call double @llvm.floor.f64(double %x)
  %fract = fsub double %x, %floor.x
  store double %fract, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fract_f64_neg:
; SI-DAG: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], -v{{\[}}[[LO]]:[[HI]]{{\]}}, -v{{\[}}[[RESLO]]:[[RESHI]]{{\]}}
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -v{{\[}}[[LO]]:[[HI]]{{\]}}, -[[SUB0]]

; CI: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; CI: v_floor_f64_e64 [[FLOORX:v\[[0-9]+:[0-9]+\]]], -[[X]]
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -[[X]], -[[FLOORX]]

; GCN-UNSAFE: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; GCN-UNSAFE: v_fract_f64_e64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -[[X]]

; GCN: buffer_store_dwordx2 [[FRACT]]
define amdgpu_kernel void @fract_f64_neg(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %neg.x = fsub double -0.0, %x
  %floor.neg.x = call double @llvm.floor.f64(double %neg.x)
  %fract = fsub double %neg.x, %floor.neg.x
  store double %fract, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fract_f64_neg_abs:
; SI-DAG: v_fract_f64_e64 [[FRC:v\[[0-9]+:[0-9]+\]]], -|v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]]|
; SI-DAG: v_mov_b32_e32 v[[UPLO:[0-9]+]], -1
; SI-DAG: v_mov_b32_e32 v[[UPHI:[0-9]+]], 0x3fefffff
; SI-DAG: v_min_f64 v{{\[}}[[MINLO:[0-9]+]]:[[MINHI:[0-9]+]]], v{{\[}}[[UPLO]]:[[UPHI]]], [[FRC]]
; SI-DAG: v_cmp_class_f64_e64 vcc, v{{\[}}[[LO]]:[[HI]]], 3
; SI: v_cndmask_b32_e32 v[[RESLO:[0-9]+]], v[[MINLO]], v[[LO]], vcc
; SI: v_cndmask_b32_e32 v[[RESHI:[0-9]+]], v[[MINHI]], v[[HI]], vcc
; SI: v_add_f64 [[SUB0:v\[[0-9]+:[0-9]+\]]], -|v{{\[}}[[LO]]:[[HI]]{{\]}}|, -v{{\[}}[[RESLO]]:[[RESHI]]{{\]}}
; SI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -|v{{\[}}[[LO]]:[[HI]]{{\]}}|, -[[SUB0]]

; CI: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; CI: v_floor_f64_e64 [[FLOORX:v\[[0-9]+:[0-9]+\]]], -|[[X]]|
; CI: v_add_f64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -|[[X]]|, -[[FLOORX]]

; GCN-UNSAFE: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; GCN-UNSAFE: v_fract_f64_e64 [[FRACT:v\[[0-9]+:[0-9]+\]]], -|[[X]]|

; GCN: buffer_store_dwordx2 [[FRACT]]
define amdgpu_kernel void @fract_f64_neg_abs(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %abs.x = call double @llvm.fabs.f64(double %x)
  %neg.abs.x = fsub double -0.0, %abs.x
  %floor.neg.abs.x = call double @llvm.floor.f64(double %neg.abs.x)
  %fract = fsub double %neg.abs.x, %floor.neg.abs.x
  store double %fract, double addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}multi_use_floor_fract_f64:
; VI-UNSAFE: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]]
; VI-UNSAFE-DAG: v_floor_f64_e32 [[FLOOR:v\[[0-9]+:[0-9]+\]]], [[X]]
; VI-UNSAFE-DAG: v_fract_f64_e32 [[FRACT:v\[[0-9]+:[0-9]+\]]], [[X]]
; VI-UNSAFE: buffer_store_dwordx2 [[FLOOR]]
; VI-UNSAFE: buffer_store_dwordx2 [[FRACT]]
define amdgpu_kernel void @multi_use_floor_fract_f64(double addrspace(1)* %out, double addrspace(1)* %src) #1 {
  %x = load double, double addrspace(1)* %src
  %floor.x = call double @llvm.floor.f64(double %x)
  %fract = fsub double %x, %floor.x
  store volatile double %floor.x, double addrspace(1)* %out
  store volatile double %fract, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
