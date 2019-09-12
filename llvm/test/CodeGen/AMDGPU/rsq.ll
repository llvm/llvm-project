; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mattr=-fp32-denormals -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.sqrt.f32(float) nounwind readnone
declare double @llvm.sqrt.f64(double) nounwind readnone

; SI-LABEL: {{^}}rsq_f32:
; SI: v_rsq_f32_e32
; SI: s_endpgm
define amdgpu_kernel void @rsq_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float, float addrspace(1)* %in, align 4
  %sqrt = call float @llvm.sqrt.f32(float %val) nounwind readnone
  %div = fdiv float 1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}rsq_f64:
; SI-UNSAFE: v_rsq_f64_e32
; SI-SAFE: v_sqrt_f64_e32
; SI: s_endpgm
define amdgpu_kernel void @rsq_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) nounwind {
  %val = load double, double addrspace(1)* %in, align 4
  %sqrt = call double @llvm.sqrt.f64(double %val) nounwind readnone
  %div = fdiv double 1.0, %sqrt
  store double %div, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}rsq_f32_sgpr:
; SI: v_rsq_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @rsq_f32_sgpr(float addrspace(1)* noalias %out, float %val) nounwind {
  %sqrt = call float @llvm.sqrt.f32(float %val) nounwind readnone
  %div = fdiv float 1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; Recognize that this is rsqrt(a) * rcp(b) * c,
; not 1 / ( 1 / sqrt(a)) * rcp(b) * c.

; SI-LABEL: @rsqrt_fmul
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-DAG: buffer_load_dword [[C:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8

; SI-UNSAFE-DAG: v_rsq_f32_e32 [[RSQA:v[0-9]+]], [[A]]
; SI-UNSAFE-DAG: v_rcp_f32_e32 [[RCPB:v[0-9]+]], [[B]]
; SI-UNSAFE-DAG: v_mul_f32_e32 [[TMP:v[0-9]+]], [[RCPB]], [[RSQA]]
; SI-UNSAFE: v_mul_f32_e32 [[RESULT:v[0-9]+]], [[C]], [[TMP]]
; SI-UNSAFE: buffer_store_dword [[RESULT]]

; SI-SAFE-NOT: v_rsq_f32

; SI: s_endpgm
define amdgpu_kernel void @rsqrt_fmul(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.2 = getelementptr float, float addrspace(1)* %gep.0, i32 2

  %a = load volatile float, float addrspace(1)* %gep.0
  %b = load volatile float, float addrspace(1)* %gep.1
  %c = load volatile float, float addrspace(1)* %gep.2

  %x = call float @llvm.sqrt.f32(float %a)
  %y = fmul float %x, %b
  %z = fdiv float %c, %y
  store float %z, float addrspace(1)* %out.gep
  ret void
}

; SI-LABEL: {{^}}neg_rsq_f32:
; SI-SAFE: v_sqrt_f32_e32 [[SQRT:v[0-9]+]], v{{[0-9]+}}
; SI-SAFE: v_rcp_f32_e64 [[RSQ:v[0-9]+]], -[[SQRT]]
; SI-SAFE: buffer_store_dword [[RSQ]]

; SI-UNSAFE: v_rsq_f32_e32 [[RSQ:v[0-9]+]], v{{[0-9]+}}
; SI-UNSAFE: v_xor_b32_e32 [[NEG_RSQ:v[0-9]+]], 0x80000000, [[RSQ]]
; SI-UNSAFE: buffer_store_dword [[NEG_RSQ]]
define amdgpu_kernel void @neg_rsq_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float, float addrspace(1)* %in, align 4
  %sqrt = call float @llvm.sqrt.f32(float %val)
  %div = fdiv float -1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}neg_rsq_f64:
; SI-SAFE: v_sqrt_f64_e32
; SI-SAFE: v_div_scale_f64

; SI-UNSAFE: v_sqrt_f64_e32 [[SQRT:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}
; SI-UNSAFE: v_rcp_f64_e64 [[RCP:v\[[0-9]+:[0-9]+\]]], -[[SQRT]]
; SI-UNSAFE: buffer_store_dwordx2 [[RCP]]
define amdgpu_kernel void @neg_rsq_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) nounwind {
  %val = load double, double addrspace(1)* %in, align 4
  %sqrt = call double @llvm.sqrt.f64(double %val)
  %div = fdiv double -1.0, %sqrt
  store double %div, double addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}neg_rsq_neg_f32:
; SI-SAFE: v_sqrt_f32_e64 [[SQRT:v[0-9]+]], -v{{[0-9]+}}
; SI-SAFE: v_rcp_f32_e64 [[RSQ:v[0-9]+]], -[[SQRT]]
; SI-SAFE: buffer_store_dword [[RSQ]]

; SI-UNSAFE: v_rsq_f32_e64 [[RSQ:v[0-9]+]], -v{{[0-9]+}}
; SI-UNSAFE: v_xor_b32_e32 [[NEG_RSQ:v[0-9]+]], 0x80000000, [[RSQ]]
; SI-UNSAFE: buffer_store_dword [[NEG_RSQ]]
define amdgpu_kernel void @neg_rsq_neg_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %val = load float, float addrspace(1)* %in, align 4
  %val.fneg = fsub float -0.0, %val
  %sqrt = call float @llvm.sqrt.f32(float %val.fneg)
  %div = fdiv float -1.0, %sqrt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}neg_rsq_neg_f64:
; SI-SAFE: v_sqrt_f64_e64 v{{\[[0-9]+:[0-9]+\]}}, -v{{\[[0-9]+:[0-9]+\]}}
; SI-SAFE: v_div_scale_f64

; SI-UNSAFE: v_sqrt_f64_e64 [[SQRT:v\[[0-9]+:[0-9]+\]]], -v{{\[[0-9]+:[0-9]+\]}}
; SI-UNSAFE: v_rcp_f64_e64 [[RCP:v\[[0-9]+:[0-9]+\]]], -[[SQRT]]
; SI-UNSAFE: buffer_store_dwordx2 [[RCP]]
define amdgpu_kernel void @neg_rsq_neg_f64(double addrspace(1)* noalias %out, double addrspace(1)* noalias %in) nounwind {
  %val = load double, double addrspace(1)* %in, align 4
  %val.fneg = fsub double -0.0, %val
  %sqrt = call double @llvm.sqrt.f64(double %val.fneg)
  %div = fdiv double -1.0, %sqrt
  store double %div, double addrspace(1)* %out, align 4
  ret void
}
