; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; SI-LABEL: {{^}}v_uint_to_fp_i64_to_f64
; SI: buffer_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; SI-DAG: v_cvt_f64_u32_e32 [[HI_CONV:v\[[0-9]+:[0-9]+\]]], v[[HI]]
; SI-DAG: v_cvt_f64_u32_e32 [[LO_CONV:v\[[0-9]+:[0-9]+\]]], v[[LO]]
; SI-DAG: v_ldexp_f64 [[LDEXP:v\[[0-9]+:[0-9]+\]]], [[HI_CONV]], 32
; SI: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[LDEXP]], [[LO_CONV]]
; SI: buffer_store_dwordx2 [[RESULT]]
define void @v_uint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep, align 8
  %result = uitofp i64 %val to double
  store double %result, double addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_i64_to_f64
define void @s_uint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 %in) {
  %cast = uitofp i64 %in to double
  store double %cast, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v2i64_to_v2f64
define void @s_uint_to_fp_v2i64_to_v2f64(<2 x double> addrspace(1)* %out, <2 x i64> %in) {
  %cast = uitofp <2 x i64> %in to <2 x double>
  store <2 x double> %cast, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v4i64_to_v4f64
define void @s_uint_to_fp_v4i64_to_v4f64(<4 x double> addrspace(1)* %out, <4 x i64> %in) {
  %cast = uitofp <4 x i64> %in to <4 x double>
  store <4 x double> %cast, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_i32_to_f64
; SI: v_cvt_f64_u32_e32
; SI: s_endpgm
define void @s_uint_to_fp_i32_to_f64(double addrspace(1)* %out, i32 %in) {
  %cast = uitofp i32 %in to double
  store double %cast, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v2i32_to_v2f64
; SI: v_cvt_f64_u32_e32
; SI: v_cvt_f64_u32_e32
; SI: s_endpgm
define void @s_uint_to_fp_v2i32_to_v2f64(<2 x double> addrspace(1)* %out, <2 x i32> %in) {
  %cast = uitofp <2 x i32> %in to <2 x double>
  store <2 x double> %cast, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}s_uint_to_fp_v4i32_to_v4f64
; SI: v_cvt_f64_u32_e32
; SI: v_cvt_f64_u32_e32
; SI: v_cvt_f64_u32_e32
; SI: v_cvt_f64_u32_e32
; SI: s_endpgm
define void @s_uint_to_fp_v4i32_to_v4f64(<4 x double> addrspace(1)* %out, <4 x i32> %in) {
  %cast = uitofp <4 x i32> %in to <4 x double>
  store <4 x double> %cast, <4 x double> addrspace(1)* %out, align 16
  ret void
}

; We can't fold the SGPRs into v_cndmask_b32_e32, because it already
; uses an SGPR (implicit vcc).

; SI-LABEL: {{^}}uint_to_fp_i1_to_f64:
; SI-DAG: v_cmp_eq_i32_e64 vcc
; SI-DAG: v_cndmask_b32_e32 v[[SEL:[0-9]+]], 0, v{{[0-9]+}}
; SI-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; SI: buffer_store_dwordx2 v{{\[}}[[ZERO]]:[[SEL]]{{\]}}
; SI: s_endpgm
define void @uint_to_fp_i1_to_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = uitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}uint_to_fp_i1_to_f64_load:
; SI: v_cndmask_b32_e64 [[IRESULT:v[0-9]]], 0, 1
; SI-NEXT: v_cvt_f64_u32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; SI: buffer_store_dwordx2 [[RESULT]]
; SI: s_endpgm
define void @uint_to_fp_i1_to_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = uitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}
