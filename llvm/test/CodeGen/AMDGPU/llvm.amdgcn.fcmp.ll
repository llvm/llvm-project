; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

declare i64 @llvm.amdgcn.fcmp.f32(float, float, i32) #0
declare i64 @llvm.amdgcn.fcmp.f64(double, double, i32) #0
declare float @llvm.fabs.f32(float) #0

declare i64 @llvm.amdgcn.fcmp.f16(half, half, i32) #0
declare half @llvm.fabs.f16(half) #0

; GCN-LABEL: {{^}}v_fcmp_f32_oeq_with_fabs:
; GCN: v_cmp_eq_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}}, |{{v[0-9]+}}|
define amdgpu_kernel void @v_fcmp_f32_oeq_with_fabs(i64 addrspace(1)* %out, float %src, float %a) {
  %temp = call float @llvm.fabs.f32(float %a)
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float %temp, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_oeq_both_operands_with_fabs:
; GCN: v_cmp_eq_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, |{{s[0-9]+}}|, |{{v[0-9]+}}|
define amdgpu_kernel void @v_fcmp_f32_oeq_both_operands_with_fabs(i64 addrspace(1)* %out, float %src, float %a) {
  %temp = call float @llvm.fabs.f32(float %a)
  %src_input = call float @llvm.fabs.f32(float %src)
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src_input, float %temp, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32:
; GCN-NOT: v_cmp_eq_f32_e64
define amdgpu_kernel void @v_fcmp_f32(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 -1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_oeq:
; GCN: v_cmp_eq_f32_e64
define amdgpu_kernel void @v_fcmp_f32_oeq(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_one:
; GCN: v_cmp_neq_f32_e64
define amdgpu_kernel void @v_fcmp_f32_one(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 6)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_ogt:
; GCN: v_cmp_gt_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ogt(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 2)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_oge:
; GCN: v_cmp_ge_f32_e64
define amdgpu_kernel void @v_fcmp_f32_oge(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 3)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_olt:
; GCN: v_cmp_lt_f32_e64
define amdgpu_kernel void @v_fcmp_f32_olt(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 4)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_ole:
; GCN: v_cmp_le_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ole(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 5)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}v_fcmp_f32_ueq:
; GCN: v_cmp_nlg_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ueq(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 9)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_une:
; GCN: v_cmp_neq_f32_e64
define amdgpu_kernel void @v_fcmp_f32_une(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 14)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_ugt:
; GCN: v_cmp_nle_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ugt(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 10)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_uge:
; GCN: v_cmp_nlt_f32_e64
define amdgpu_kernel void @v_fcmp_f32_uge(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 11)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_ult:
; GCN: v_cmp_nge_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ult(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 12)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f32_ule:
; GCN: v_cmp_ngt_f32_e64
define amdgpu_kernel void @v_fcmp_f32_ule(i64 addrspace(1)* %out, float %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f32(float %src, float 100.00, i32 13)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_oeq:
; GCN: v_cmp_eq_f64_e64
define amdgpu_kernel void @v_fcmp_f64_oeq(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_one:
; GCN: v_cmp_neq_f64_e64
define amdgpu_kernel void @v_fcmp_f64_one(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 6)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ogt:
; GCN: v_cmp_gt_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ogt(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 2)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_oge:
; GCN: v_cmp_ge_f64_e64
define amdgpu_kernel void @v_fcmp_f64_oge(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 3)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_olt:
; GCN: v_cmp_lt_f64_e64
define amdgpu_kernel void @v_fcmp_f64_olt(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 4)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ole:
; GCN: v_cmp_le_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ole(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 5)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ueq:
; GCN: v_cmp_nlg_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ueq(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 9)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_une:
; GCN: v_cmp_neq_f64_e64
define amdgpu_kernel void @v_fcmp_f64_une(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 14)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ugt:
; GCN: v_cmp_nle_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ugt(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 10)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_uge:
; GCN: v_cmp_nlt_f64_e64
define amdgpu_kernel void @v_fcmp_f64_uge(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 11)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ult:
; GCN: v_cmp_nge_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ult(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 12)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f64_ule:
; GCN: v_cmp_ngt_f64_e64
define amdgpu_kernel void @v_fcmp_f64_ule(i64 addrspace(1)* %out, double %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f64(double %src, double 100.00, i32 13)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_oeq_with_fabs:
; VI: v_cmp_eq_f16_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}}, |{{v[0-9]+}}|

; SI: v_cvt_f32_f16_e32 [[CVT0:v[0-9]+]], s{{[0-9]+}}
; SI: v_cvt_f32_f16_e64 [[CVT1:v[0-9]+]], |s{{[0-9]+}}|
; SI: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT0]], [[CVT1]]
define amdgpu_kernel void @v_fcmp_f16_oeq_with_fabs(i64 addrspace(1)* %out, half %src, half %a) {
  %temp = call half @llvm.fabs.f16(half %a)
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half %temp, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_oeq_both_operands_with_fabs:
; VI: v_cmp_eq_f16_e64 {{s\[[0-9]+:[0-9]+\]}}, |{{s[0-9]+}}|, |{{v[0-9]+}}|

; SI: v_cvt_f32_f16_e64 [[CVT0:v[0-9]+]], |s{{[0-9]+}}|
; SI: v_cvt_f32_f16_e64 [[CVT1:v[0-9]+]], |s{{[0-9]+}}|
; SI: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[CVT0]], [[CVT1]]
define amdgpu_kernel void @v_fcmp_f16_oeq_both_operands_with_fabs(i64 addrspace(1)* %out, half %src, half %a) {
  %temp = call half @llvm.fabs.f16(half %a)
  %src_input = call half @llvm.fabs.f16(half %src)
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src_input, half %temp, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16:
; GCN-NOT: v_cmp_eq_
define amdgpu_kernel void @v_fcmp_f16(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 -1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_oeq:
; VI: v_cmp_eq_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_oeq(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 1)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_one:
; VI: v_cmp_neq_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_neq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_one(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 6)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ogt:
; VI: v_cmp_gt_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_lt_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ogt(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 2)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_oge:
; VI: v_cmp_ge_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_le_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_oge(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 3)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_olt:
; VI: v_cmp_lt_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_gt_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_olt(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 4)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ole:
; VI: v_cmp_le_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_ge_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ole(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 5)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ueq:
; VI: v_cmp_nlg_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_nlg_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ueq(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 9)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_une:
; VI: v_cmp_neq_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_neq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_une(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 14)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ugt:
; VI: v_cmp_nle_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_nge_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ugt(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 10)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_uge:
; VI: v_cmp_nlt_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_ngt_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_uge(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 11)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ult:
; VI: v_cmp_nge_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_nle_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ult(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 12)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fcmp_f16_ule:
; VI: v_cmp_ngt_f16_e64

; SI-DAG: s_mov_b32 [[K:s[0-9]+]], 0x42c80000
; SI-DAG: v_cvt_f32_f16_e32 [[CVT:v[0-9]+]], s{{[0-9]+}}
; SI: v_cmp_nlt_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[CVT]]
define amdgpu_kernel void @v_fcmp_f16_ule(i64 addrspace(1)* %out, half %src) {
  %result = call i64 @llvm.amdgcn.fcmp.f16(half %src, half 100.00, i32 13)
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone convergent }
