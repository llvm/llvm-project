; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; SI-LABEL: {{^}}sint_to_fp_i32_to_f64
; SI: v_cvt_f64_i32_e32
define void @sint_to_fp_i32_to_f64(double addrspace(1)* %out, i32 %in) {
  %result = sitofp i32 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; FIXME: select on 0, 0
; SI-LABEL: {{^}}sint_to_fp_i1_f64:
; SI: v_cmp_eq_i32_e64 vcc,
; We can't fold the SGPRs into v_cndmask_b32_e64, because it already
; uses an SGPR (implicit vcc).
; SI: v_cndmask_b32_e32 v{{[0-9]+}}, 0, v{{[0-9]+}}
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 0, vcc
; SI: buffer_store_dwordx2
; SI: s_endpgm
define void @sint_to_fp_i1_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = sitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}sint_to_fp_i1_f64_load:
; SI: v_cndmask_b32_e64 [[IRESULT:v[0-9]]], 0, -1
; SI-NEXT: v_cvt_f64_i32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: buffer_store_dwordx2 [[RESULT]]
; SI: s_endpgm
define void @sint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = sitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: @s_sint_to_fp_i64_to_f64
define void @s_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 %in) {
  %result = sitofp i64 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: @v_sint_to_fp_i64_to_f64
; SI: buffer_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; SI-DAG: v_cvt_f64_i32_e32 [[HI_CONV:v\[[0-9]+:[0-9]+\]]], v[[HI]]
; SI-DAG: v_cvt_f64_u32_e32 [[LO_CONV:v\[[0-9]+:[0-9]+\]]], v[[LO]]
; SI-DAG: v_ldexp_f64 [[LDEXP:v\[[0-9]+:[0-9]+\]]], [[HI_CONV]], 32
; SI: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[LDEXP]], [[LO_CONV]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @v_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep, align 8
  %result = sitofp i64 %val to double
  store double %result, double addrspace(1)* %out
  ret void
}
