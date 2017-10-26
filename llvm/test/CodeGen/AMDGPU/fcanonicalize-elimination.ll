; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=-fp32-denormals < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=GCN-FLUSH %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=-fp32-denormals,+fp-exceptions < %s | FileCheck -check-prefix=GCN -check-prefix=GCN-EXCEPT -check-prefix=VI -check-prefix=GCN-FLUSH %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -verify-machineinstrs -mattr=+fp32-denormals < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX9-DENORM %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -verify-machineinstrs -mattr=-fp32-denormals < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GCN-FLUSH %s

; GCN-LABEL: {{^}}test_no_fold_canonicalize_loaded_value_f32:
; GCN-FLUSH:   v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; GFX9-DENORM: v_max_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_no_fold_canonicalize_loaded_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %v = load float, float addrspace(1)* %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_fmul_value_f32:
; GCN: v_mul_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fmul_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fmul float %load, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_sub_value_f32:
; GCN: v_sub_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_sub_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fsub float 15.0, %load
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_add_value_f32:
; GCN: v_add_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_add_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fadd float %load, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_sqrt_value_f32:
; GCN: v_sqrt_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_sqrt_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.sqrt.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fceil_value_f32:
; GCN: v_ceil_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fceil_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.ceil.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_floor_value_f32:
; GCN: v_floor_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_floor_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.floor.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fma_value_f32:
; GCN: v_fma_f32 [[V:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fma_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.fma.f32(float %load, float 15.0, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fmuladd_value_f32:
; GCN-FLUSH: v_mac_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; GFX9-DENORM: v_fma_f32 [[V:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fmuladd_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.fmuladd.f32(float %load, float 15.0, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_canonicalize_value_f32:
; GCN: {{flat|global}}_load_dword [[LOAD:v[0-9]+]],
; GCN-FLUSH:  v_mul_f32_e32 [[V:v[0-9]+]], 1.0, [[LOAD]]
; GCN-DENORM: v_max_f32_e32 [[V:v[0-9]+]], [[LOAD]], [[LOAD]]
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_canonicalize_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = call float @llvm.canonicalize.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpextend_value_f64_f32:
; GCN: v_cvt_f64_f32_e32 [[V:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dwordx2 v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fpextend_value_f64_f32(float addrspace(1)* %arg, double addrspace(1)* %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fpext float %load to double
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, double addrspace(1)* %out, i32 %id
  store double %canonicalized, double addrspace(1)* %gep2, align 8
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpextend_value_f32_f16:
; GCN: v_cvt_f32_f16_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fpextend_value_f32_f16(half addrspace(1)* %arg, float addrspace(1)* %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %load = load half, half addrspace(1)* %gep, align 2
  %v = fpext half %load to float
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i32 %id
  store float %canonicalized, float addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_f32_f64:
; GCN: v_cvt_f32_f64_e32 [[V:v[0-9]+]], v[{{[0-9:]+}}]
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_f32_f64(double addrspace(1)* %arg, float addrspace(1)* %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, double addrspace(1)* %arg, i32 %id
  %load = load double, double addrspace(1)* %gep, align 8
  %v = fptrunc double %load to float
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i32 %id
  store float %canonicalized, float addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_f16_f32:
; GCN: v_cvt_f16_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_short v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_f16_f32(float addrspace(1)* %arg, half addrspace(1)* %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fptrunc float %load to half
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, half addrspace(1)* %out, i32 %id
  store half %canonicalized, half addrspace(1)* %gep2, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_v2f16_v2f32:
; GCN-DAG: v_cvt_f16_f32_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; VI-DAG: v_cvt_f16_f32_sdwa [[V1:v[0-9]+]], v{{[0-9]+}}
; VI: v_or_b32_e32 [[V:v[0-9]+]], [[V0]], [[V1]]
; GFX9: v_cvt_f16_f32_e32 [[V1:v[0-9]+]], v{{[0-9]+}}
; GFX9: v_and_b32_e32 [[V0_16:v[0-9]+]], 0xffff, [[V0]]
; GFX9: v_lshl_or_b32 [[V:v[0-9]+]], [[V1]], 16, [[V0_16]]
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_v2f16_v2f32(<2 x float> addrspace(1)* %arg, <2 x half> addrspace(1)* %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, <2 x float> addrspace(1)* %arg, i32 %id
  %load = load <2 x float>, <2 x float> addrspace(1)* %gep, align 8
  %v = fptrunc <2 x float> %load to <2 x half>
  %canonicalized = tail call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %v)
  %gep2 = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i32 %id
  store <2 x half> %canonicalized, <2 x half> addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fneg_value_f32:
; GCN-FLUSH:  v_mul_f32_e64 v{{[0-9]+}}, 1.0, -v{{[0-9]+}}
; GCN-DENORM: v_max_f32_e64 v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
define amdgpu_kernel void @test_no_fold_canonicalize_fneg_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = fsub float -0.0, %load
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fneg_value_f32:
; GCN: v_xor_b32_e32 [[V:v[0-9]+]], 0x80000000, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fneg_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = fsub float -0.0, %v0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fabs_value_f32:
; GCN-FLUSH:  v_mul_f32_e64 v{{[0-9]+}}, 1.0, |v{{[0-9]+}}|
; GCN-DENORM: v_max_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, |v{{[0-9]+}}|
define amdgpu_kernel void @test_no_fold_canonicalize_fabs_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.fabs.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fabs_value_f32:
; GCN: v_and_b32_e32 [[V:v[0-9]+]], 0x7fffffff, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fabs_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.fabs.f32(float %v0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_sin_value_f32:
; GCN: v_sin_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_sin_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.sin.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_cos_value_f32:
; GCN: v_cos_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_cos_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.cos.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_sin_value_f16:
; GCN: v_sin_f32_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; GCN: v_cvt_f16_f32_e32 [[V:v[0-9]+]], [[V0]]
; GCN: {{flat|global}}_store_short v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_sin_value_f16(half addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %load = load half, half addrspace(1)* %gep, align 2
  %v = tail call half @llvm.sin.f16(half %load)
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  store half %canonicalized, half addrspace(1)* %gep, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_cos_value_f16:
; GCN: v_cos_f32_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; GCN: v_cvt_f16_f32_e32 [[V:v[0-9]+]], [[V0]]
; GCN: {{flat|global}}_store_short v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_cos_value_f16(half addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %load = load half, half addrspace(1)* %gep, align 2
  %v = tail call half @llvm.cos.f16(half %load)
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  store half %canonicalized, half addrspace(1)* %gep, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_qNaN_value_f32:
; GCN: v_mov_b32_e32 [[V:v[0-9]+]], 0x7fc00000
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_qNaN_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %canonicalized = tail call float @llvm.canonicalize.f32(float 0x7FF8000000000000)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_minnum_value_from_load_f32:
; VI: v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; GFX9: v_min_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; GFX9: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
define amdgpu_kernel void @test_fold_canonicalize_minnum_value_from_load_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.minnum.f32(float %load, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_minnum_value_f32:
; GCN: v_min_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_minnum_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.minnum.f32(float %v0, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_sNaN_value_f32:
; GCN:  v_min_f32_e32 [[V0:v[0-9]+]], 0x7f800001, v{{[0-9]+}}
; GCN-FLUSH:  v_mul_f32_e32 v{{[0-9]+}}, 1.0, [[V0]]
; GCN-DENORM: v_max_f32_e32 v{{[0-9]+}}, [[V0]], [[V0]]
; GCN:  {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
define amdgpu_kernel void @test_fold_canonicalize_sNaN_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.minnum.f32(float %load, float bitcast (i32 2139095041 to float))
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_denorm_value_f32:
; GFX9:  v_min_f32_e32 [[V:v[0-9]+]], 0x7fffff, v{{[0-9]+}}
; VI:    v_min_f32_e32 [[V0:v[0-9]+]], 0x7fffff, v{{[0-9]+}}
; VI:    v_mul_f32_e32 v{{[0-9]+}}, 1.0, [[V0]]
; GCN:   {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GFX9-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_denorm_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.minnum.f32(float %load, float bitcast (i32 8388607 to float))
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_maxnum_value_from_load_f32:
; GFX9:  v_max_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; VI:    v_max_f32_e32 [[V0:v[0-9]+]], 0, v{{[0-9]+}}
; VI:    v_mul_f32_e32 v{{[0-9]+}}, 1.0, [[V0]]
; GCN:  {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GFX9-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_from_load_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v = tail call float @llvm.maxnum.f32(float %load, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_maxnum_value_f32:
; GCN: v_max_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_f32(float addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %load = load float, float addrspace(1)* %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.maxnum.f32(float %v0, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, float addrspace(1)* %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_maxnum_value_f64:
; GCN: v_max_f64 [[V:v\[[0-9]+:[0-9]+\]]], v[{{[0-9:]+}}], 0
; GCN: {{flat|global}}_store_dwordx2 v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_f64(double addrspace(1)* %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, double addrspace(1)* %arg, i32 %id
  %load = load double, double addrspace(1)* %gep, align 8
  %v0 = fadd double %load, 0.0
  %v = tail call double @llvm.maxnum.f64(double %v0, double 0.0)
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  store double %canonicalized, double addrspace(1)* %gep, align 8
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fmul_value_f32_no_ieee:
; GCN-EXCEPT: v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
define amdgpu_ps float @test_no_fold_canonicalize_fmul_value_f32_no_ieee(float %arg) {
entry:
  %v = fmul float %arg, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: test_fold_canonicalize_fmul_nnan_value_f32_no_ieee:
; GCN: v_mul_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NEXT: ; return
; GCN-NOT: 1.0
define amdgpu_ps float @test_fold_canonicalize_fmul_nnan_value_f32_no_ieee(float %arg) {
entry:
  %v = fmul nnan float %arg, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f32
; GFX9-DENORM: global_load_dword [[V:v[0-9]+]],
; GFX9-DENORM: global_store_dword v[{{[0-9:]+}}], [[V]]
; GFX9-DENORM-NOT: 1.0
; GCN-FLUSH: v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f32(float addrspace(1)* %arg, float addrspace(1)* %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, float addrspace(1)* %arg, i32 %id
  %v = load float, float addrspace(1)* %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i32 %id
  store float %canonicalized, float addrspace(1)* %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f64
; GCN: {{flat|global}}_load_dwordx2 [[V:v\[[0-9:]+\]]],
; GCN: {{flat|global}}_store_dwordx2 v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f64(double addrspace(1)* %arg, double addrspace(1)* %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, double addrspace(1)* %arg, i32 %id
  %v = load double, double addrspace(1)* %gep, align 8
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, double addrspace(1)* %out, i32 %id
  store double %canonicalized, double addrspace(1)* %gep2, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f16
; GCN: {{flat|global}}_load_ushort [[V:v[0-9]+]],
; GCN: {{flat|global}}_store_short v[{{[0-9:]+}}], [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f16(half addrspace(1)* %arg, half addrspace(1)* %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, half addrspace(1)* %arg, i32 %id
  %v = load half, half addrspace(1)* %gep, align 2
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, half addrspace(1)* %out, i32 %id
  store half %canonicalized, half addrspace(1)* %gep2, align 2
  ret void
}

; Avoid failing the test on FreeBSD11.0 which will match the GCN-NOT: 1.0
; in the .amd_amdgpu_isa "amdgcn-unknown-freebsd11.0--gfx802" directive
; CHECK: .amd_amdgpu_isa

declare float @llvm.canonicalize.f32(float) #0
declare double @llvm.canonicalize.f64(double) #0
declare half @llvm.canonicalize.f16(half) #0
declare <2 x half> @llvm.canonicalize.v2f16(<2 x half>) #0
declare i32 @llvm.amdgcn.workitem.id.x() #0
declare float @llvm.sqrt.f32(float) #0
declare float @llvm.ceil.f32(float) #0
declare float @llvm.floor.f32(float) #0
declare float @llvm.fma.f32(float, float, float) #0
declare float @llvm.fmuladd.f32(float, float, float) #0
declare float @llvm.fabs.f32(float) #0
declare float @llvm.sin.f32(float) #0
declare float @llvm.cos.f32(float) #0
declare half @llvm.sin.f16(half) #0
declare half @llvm.cos.f16(half) #0
declare float @llvm.minnum.f32(float, float) #0
declare float @llvm.maxnum.f32(float, float) #0
declare double @llvm.maxnum.f64(double, double) #0

attributes #0 = { nounwind readnone }
attributes #1 = { "no-nans-fp-math"="true" }
