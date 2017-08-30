; RUN:  llc -amdgpu-scalarize-global-loads=false -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.fabs.f32(float) #0
declare float @llvm.canonicalize.f32(float) #0
declare double @llvm.fabs.f64(double) #0
declare double @llvm.canonicalize.f64(double) #0
declare half @llvm.canonicalize.f16(half) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0

; GCN-LABEL: {{^}}v_test_canonicalize_var_f32:
; GCN: v_mul_f32_e32 [[REG:v[0-9]+]], 1.0, {{v[0-9]+}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @v_test_canonicalize_var_f32(float addrspace(1)* %out) #1 {
  %val = load float, float addrspace(1)* %out
  %canonicalized = call float @llvm.canonicalize.f32(float %val)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_f32:
; GCN: v_mul_f32_e64 [[REG:v[0-9]+]], 1.0, {{s[0-9]+}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_f32(float addrspace(1)* %out, float %val) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float %val)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_f32:
; GCN: v_mul_f32_e64 [[REG:v[0-9]+]], 1.0, |{{v[0-9]+}}|
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fabs_var_f32(float addrspace(1)* %out) #1 {
  %val = load float, float addrspace(1)* %out
  %val.fabs = call float @llvm.fabs.f32(float %val)
  %canonicalized = call float @llvm.canonicalize.f32(float %val.fabs)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_fabs_var_f32:
; GCN: v_mul_f32_e64 [[REG:v[0-9]+]], 1.0, -|{{v[0-9]+}}|
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_f32(float addrspace(1)* %out) #1 {
  %val = load float, float addrspace(1)* %out
  %val.fabs = call float @llvm.fabs.f32(float %val)
  %val.fabs.fneg = fsub float -0.0, %val.fabs
  %canonicalized = call float @llvm.canonicalize.f32(float %val.fabs.fneg)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_var_f32:
; GCN: v_mul_f32_e64 [[REG:v[0-9]+]], 1.0, -{{v[0-9]+}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_var_f32(float addrspace(1)* %out) #1 {
  %val = load float, float addrspace(1)* %out
  %val.fneg = fsub float -0.0, %val
  %canonicalized = call float @llvm.canonicalize.f32(float %val.fneg)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p0_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float 0.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_f32:
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n0_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float -0.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_p1_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float 1.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], -1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_n1_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float -1.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x41800000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_literal_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float 16.0)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal0_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 8388607 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fffff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_f32(float addrspace(1)* %out) #3 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 8388607 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal1_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal1_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2155872255 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x807fffff{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_f32(float addrspace(1)* %out) #3 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2155872255 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float 0x7FF8000000000000)
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 -1 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 -2 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2139095041 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 2143289343 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 4286578689 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x7fc00000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_f32(float addrspace(1)* %out) #1 {
  %canonicalized = call float @llvm.canonicalize.f32(float bitcast (i32 4290772991 to float))
  store float %canonicalized, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_var_f64:
; GCN: v_max_f64 [[REG:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @v_test_canonicalize_var_f64(double addrspace(1)* %out) #1 {
  %val = load double, double addrspace(1)* %out
  %canonicalized = call double @llvm.canonicalize.f64(double %val)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_test_canonicalize_var_f64:
; GCN: v_max_f64 [[REG:v\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @s_test_canonicalize_var_f64(double addrspace(1)* %out, double %val) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double %val)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fabs_var_f64:
; GCN: v_max_f64 [[REG:v\[[0-9]+:[0-9]+\]]], |{{v\[[0-9]+:[0-9]+\]}}|, |{{v\[[0-9]+:[0-9]+\]}}|
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fabs_var_f64(double addrspace(1)* %out) #1 {
  %val = load double, double addrspace(1)* %out
  %val.fabs = call double @llvm.fabs.f64(double %val)
  %canonicalized = call double @llvm.canonicalize.f64(double %val.fabs)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_fabs_var_f64:
; GCN: v_max_f64 [[REG:v\[[0-9]+:[0-9]\]]], -|{{v\[[0-9]+:[0-9]+\]}}|, -|{{v\[[0-9]+:[0-9]+\]}}|
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_fabs_var_f64(double addrspace(1)* %out) #1 {
  %val = load double, double addrspace(1)* %out
  %val.fabs = call double @llvm.fabs.f64(double %val)
  %val.fabs.fneg = fsub double -0.0, %val.fabs
  %canonicalized = call double @llvm.canonicalize.f64(double %val.fabs.fneg)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_test_canonicalize_fneg_var_f64:
; GCN: v_max_f64 [[REG:v\[[0-9]+:[0-9]+\]]], -{{v\[[0-9]+:[0-9]+\]}}, -{{v\[[0-9]+:[0-9]+\]}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @v_test_canonicalize_fneg_var_f64(double addrspace(1)* %out) #1 {
  %val = load double, double addrspace(1)* %out
  %val.fneg = fsub double -0.0, %val
  %canonicalized = call double @llvm.canonicalize.f64(double %val.fneg)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p0_f64:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], v[[LO]]{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_p0_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double 0.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HI:[0-9]+]], 1{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_n0_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double -0.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_p1_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x3ff00000{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_p1_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double 1.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_n1_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0xbff00000{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_n1_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double -1.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_literal_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x40300000{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_literal_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double 16.0)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal0_f64:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], v[[LO]]{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal0_f64(double addrspace(1)* %out) #2 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 4503599627370495 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], -1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0xfffff{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal0_f64(double addrspace(1)* %out) #3 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 4503599627370495 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_no_denormals_fold_canonicalize_denormal1_f64:
; GCN: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[HI:[0-9]+]], v[[LO]]{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_no_denormals_fold_canonicalize_denormal1_f64(double addrspace(1)* %out) #2 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9227875636482146303 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_denormals_fold_canonicalize_denormal1_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], -1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x800fffff{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_denormals_fold_canonicalize_denormal1_f64(double addrspace(1)* %out) #3 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9227875636482146303 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_qnan_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double 0x7FF8000000000000)
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg1_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg1_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 -1 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_qnan_value_neg2_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_qnan_value_neg2_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 -2 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan0_value_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_snan0_value_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9218868437227405313 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan1_value_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_snan1_value_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 9223372036854775807 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan2_value_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_snan2_value_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 18442240474082181121 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_snan3_value_f64:
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], 0x7ff80000{{$}}
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @test_fold_canonicalize_snan3_value_f64(double addrspace(1)* %out) #1 {
  %canonicalized = call double @llvm.canonicalize.f64(double bitcast (i64 18446744073709551615 to double))
  store double %canonicalized, double addrspace(1)* %out
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f64_flush:
; GCN: v_mul_f64 v[{{[0-9:]+}}], 1.0, v[{{[0-9:]+}}]
define amdgpu_kernel void @test_canonicalize_value_f64_flush(double addrspace(1)* %arg, double addrspace(1)* %out) #4 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, double addrspace(1)* %arg, i32 %id
  %v = load double, double addrspace(1)* %gep, align 8
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, double addrspace(1)* %out, i32 %id
  store double %canonicalized, double addrspace(1)* %gep2, align 8
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f32_flush:
; GCN: v_mul_f32_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @test_canonicalize_value_f32_flush(float addrspace(1)* %arg, float addrspace(1)* %out) #4 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %v = load float, float addrspace(1)* %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i32 %id
  store float %canonicalized, float addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f16_flush:
; GCN: v_mul_f16_e32 {{v[0-9]+}}, 1.0, {{v[0-9]+}}
define amdgpu_kernel void @test_canonicalize_value_f16_flush(half addrspace(1)* %arg, half addrspace(1)* %out) #4 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %v = load half, half addrspace(1)* %gep, align 2
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, half addrspace(1)* %out, i32 %id
  store half %canonicalized, half addrspace(1)* %gep2, align 2
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f64_denorm:
; GCN: v_max_f64 v[{{[0-9:]+}}], v[{{[0-9:]+}}], v[{{[0-9:]+}}]
define amdgpu_kernel void @test_canonicalize_value_f64_denorm(double addrspace(1)* %arg, double addrspace(1)* %out) #5 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, double addrspace(1)* %arg, i32 %id
  %v = load double, double addrspace(1)* %gep, align 8
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, double addrspace(1)* %out, i32 %id
  store double %canonicalized, double addrspace(1)* %gep2, align 8
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f32_denorm:
; GCN: v_max_f32_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @test_canonicalize_value_f32_denorm(float addrspace(1)* %arg, float addrspace(1)* %out) #5 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %v = load float, float addrspace(1)* %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i32 %id
  store float %canonicalized, float addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL:  {{^}}test_canonicalize_value_f16_denorm:
; GCN: v_max_f16_e32 {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
define amdgpu_kernel void @test_canonicalize_value_f16_denorm(half addrspace(1)* %arg, half addrspace(1)* %out) #5 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %v = load half, half addrspace(1)* %gep, align 2
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, half addrspace(1)* %out, i32 %id
  store half %canonicalized, half addrspace(1)* %gep2, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind "target-features"="-fp32-denormals,-fp64-fp16-denormals" }
attributes #3 = { nounwind "target-features"="+fp32-denormals,+fp64-fp16-denormals" }
attributes #4 = { nounwind "target-features"="-fp32-denormals,-fp64-fp16-denormals" "target-cpu"="tonga" }
attributes #5 = { nounwind "target-features"="+fp32-denormals,+fp64-fp16-denormals" "target-cpu"="gfx900" }
