; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; GCN-LABEL: {{^}}sint_to_fp_i32_to_f64
; GCN: v_cvt_f64_i32_e32
define amdgpu_kernel void @sint_to_fp_i32_to_f64(double addrspace(1)* %out, i32 %in) {
  %result = sitofp i32 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; We can't fold the SGPRs into v_cndmask_b32_e64, because it already
; uses an SGPR (implicit vcc).

; GCN-LABEL: {{^}}sint_to_fp_i1_f64:
; GCN-DAG: v_cmp_eq_u32_e64 vcc,
; GCN-DAG: v_cndmask_b32_e32 v[[SEL:[0-9]+]], 0, v{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 v[[ZERO:[0-9]+]], 0{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[ZERO]]:[[SEL]]{{\]}}
; GCN: s_endpgm
define amdgpu_kernel void @sint_to_fp_i1_f64(double addrspace(1)* %out, i32 %in) {
  %cmp = icmp eq i32 %in, 0
  %fp = sitofp i1 %cmp to double
  store double %fp, double addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}sint_to_fp_i1_f64_load:
; GCN: v_cndmask_b32_e64 [[IRESULT:v[0-9]]], 0, -1
; GCN: v_cvt_f64_i32_e32 [[RESULT:v\[[0-9]+:[0-9]\]]], [[IRESULT]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @sint_to_fp_i1_f64_load(double addrspace(1)* %out, i1 %in) {
  %fp = sitofp i1 %in to double
  store double %fp, double addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: @s_sint_to_fp_i64_to_f64
define amdgpu_kernel void @s_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 %in) {
  %result = sitofp i64 %in to double
  store double %result, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: @v_sint_to_fp_i64_to_f64
; GCN: flat_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN-DAG: v_cvt_f64_i32_e32 [[HI_CONV:v\[[0-9]+:[0-9]+\]]], v[[HI]]
; GCN-DAG: v_cvt_f64_u32_e32 [[LO_CONV:v\[[0-9]+:[0-9]+\]]], v[[LO]]
; GCN-DAG: v_ldexp_f64 [[LDEXP:v\[[0-9]+:[0-9]+\]]], [[HI_CONV]], 32
; GCN: v_add_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[LDEXP]], [[LO_CONV]]
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_sint_to_fp_i64_to_f64(double addrspace(1)* %out, i64 addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %val = load i64, i64 addrspace(1)* %gep, align 8
  %result = sitofp i64 %val to double
  store double %result, double addrspace(1)* %out
  ret void
}

; FIXME: bfe and sext on VI+
; GCN-LABEL: {{^}}s_sint_to_fp_i8_to_f64:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; SI-NOT: bfe
; SI: s_sext_i32_i8 [[SEXT:s[0-9]+]], [[VAL]]

; VI: s_bfe_i32 [[BFE:s[0-9]+]], [[VAL]], 0x80000
; VI: s_sext_i32_i16 [[SEXT:s[0-9]+]], [[BFE]]

; GCN: v_cvt_f64_i32_e32 v{{\[[0-9]+:[0-9]+\]}}, [[SEXT]]
define amdgpu_kernel void @s_sint_to_fp_i8_to_f64(double addrspace(1)* %out, i8 %in) {
  %fp = sitofp i8 %in to double
  store double %fp, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_sint_to_fp_i8_to_f64:
; GCN: v_bfe_i32 [[SEXT:v[0-9]+]]
; GCN: v_cvt_f64_i32_e32 v{{\[[0-9]+:[0-9]+\]}}, [[SEXT]]
define double @v_sint_to_fp_i8_to_f64(i8 %in) {
  %fp = sitofp i8 %in to double
  ret double %fp
}
