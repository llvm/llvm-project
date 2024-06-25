; RUN: llc -mtriple=amdgcn -mcpu=gfx801 -verify-machineinstrs -denormal-fp-math-f32=preserve-sign < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,VI-FLUSH,GCN-FLUSH %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx801 -verify-machineinstrs -denormal-fp-math-f32=ieee < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,VI-DENORM,GCN-DENORM %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -denormal-fp-math-f32=ieee < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-DENORM,GCN-DENORM %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs -denormal-fp-math-f32=preserve-sign < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX9-FLUSH,GCN-FLUSH %s

; GCN-LABEL: {{^}}test_no_fold_canonicalize_loaded_value_f32:
; VI: v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; GFX9: v_max_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_no_fold_canonicalize_loaded_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %v = load float, ptr addrspace(1) %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_fmul_value_f32:
; GCN: v_mul_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fmul_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fmul float %load, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_fmul_legacy_value_f32:
; GCN: v_mul_legacy_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fmul_legacy_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.amdgcn.fmul.legacy(float %load, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_sub_value_f32:
; GCN: v_sub_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_sub_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fsub float 15.0, %load
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_add_value_f32:
; GCN: v_add_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_add_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fadd float %load, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_sqrt_value_f32:
; GCN: v_sqrt_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_sqrt_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call afn float @llvm.sqrt.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fceil_value_f32:
; GCN: v_ceil_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fceil_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.ceil.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_floor_value_f32:
; GCN: v_floor_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_floor_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.floor.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fma_value_f32:
; GCN: s_mov_b32 [[SREG:s[0-9]+]], 0x41700000
; GCN: v_fma_f32 [[V:v[0-9]+]], v{{[0-9]+}}, [[SREG]], [[SREG]]
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fma_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.fma.f32(float %load, float 15.0, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fmad_ftz_value_f32:
; GCN: v_mac_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+$}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fmad_ftz_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.amdgcn.fmad.ftz.f32(float %load, float 15.0, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fmuladd_value_f32:
; GCN-FLUSH: v_mac_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-DENORM: s_mov_b32 [[SREG:s[0-9]+]], 0x41700000
; GCN-DENORM: v_fma_f32 [[V:v[0-9]+]], v{{[0-9]+}}, [[SREG]], [[SREG]]
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
; GCN-NOT: 1.0
define amdgpu_kernel void @test_fold_canonicalize_fmuladd_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.fmuladd.f32(float %load, float 15.0, float 15.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_canonicalize_value_f32:
; GCN: {{flat|global}}_load_dword [[LOAD:v[0-9]+]],
; VI:  v_mul_f32_e32 [[V:v[0-9]+]], 1.0, [[LOAD]]
; GFX9: v_max_f32_e32 [[V:v[0-9]+]], [[LOAD]], [[LOAD]]

; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_canonicalize_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = call float @llvm.canonicalize.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpextend_value_f64_f32:
; GCN: v_cvt_f64_f32_e32 [[V:v\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dwordx2 v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpextend_value_f64_f32(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fpext float %load to double
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, ptr addrspace(1) %out, i32 %id
  store double %canonicalized, ptr addrspace(1) %gep2, align 8
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpextend_value_f32_f16:
; GCN: v_cvt_f32_f16_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpextend_value_f32_f16(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, ptr addrspace(1) %arg, i32 %id
  %load = load half, ptr addrspace(1) %gep, align 2
  %v = fpext half %load to float
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %id
  store float %canonicalized, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpextend_value_f32_f16_flushf16:
; GCN: v_cvt_f32_f16_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpextend_value_f32_f16_flushf16(ptr addrspace(1) %arg, ptr addrspace(1) %out) #2 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, ptr addrspace(1) %arg, i32 %id
  %load = load half, ptr addrspace(1) %gep, align 2
  %v = fpext half %load to float
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %id
  store float %canonicalized, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_f32_f64:
; GCN: v_cvt_f32_f64_e32 [[V:v[0-9]+]], v[{{[0-9:]+}}]
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_f32_f64(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, ptr addrspace(1) %arg, i32 %id
  %load = load double, ptr addrspace(1) %gep, align 8
  %v = fptrunc double %load to float
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %id
  store float %canonicalized, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_f16_f32:
; GCN: v_cvt_f16_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_max
; GCN-NOT: v_mul
; GCN: {{flat|global}}_store_short v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_f16_f32(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fptrunc float %load to half
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, ptr addrspace(1) %out, i32 %id
  store half %canonicalized, ptr addrspace(1) %gep2, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_f16_f32_flushf16:
; GCN: v_cvt_f16_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_max
; GCN-NOT: v_mul
; GCN: {{flat|global}}_store_short v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_f16_f32_flushf16(ptr addrspace(1) %arg, ptr addrspace(1) %out) #2 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fptrunc float %load to half
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, ptr addrspace(1) %out, i32 %id
  store half %canonicalized, ptr addrspace(1) %gep2, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fpround_value_v2f16_v2f32:
; GCN-DAG: v_cvt_f16_f32_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; VI-DAG: v_cvt_f16_f32_sdwa [[V1:v[0-9]+]], v{{[0-9]+}}
; VI: v_or_b32_e32 [[V:v[0-9]+]], [[V0]], [[V1]]
; GFX9: v_cvt_f16_f32_e32 [[V1:v[0-9]+]], v{{[0-9]+}}
; GFX9: v_pack_b32_f16 [[V:v[0-9]+]], [[V1]], [[V0]]
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fpround_value_v2f16_v2f32(ptr addrspace(1) %arg, ptr addrspace(1) %out) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <2 x float>, ptr addrspace(1) %arg, i32 %id
  %load = load <2 x float>, ptr addrspace(1) %gep, align 8
  %v = fptrunc <2 x float> %load to <2 x half>
  %canonicalized = tail call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %v)
  %gep2 = getelementptr inbounds <2 x half>, ptr addrspace(1) %out, i32 %id
  store <2 x half> %canonicalized, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fneg_value_f32:
; VI:  v_mul_f32_e32 v{{[0-9]+}}, -1.0, v{{[0-9]+}}
; GFX9: v_max_f32_e64 v{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
define amdgpu_kernel void @test_no_fold_canonicalize_fneg_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = fneg float %load
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fneg_value_f32:
; GCN: v_xor_b32_e32 [[V:v[0-9]+]], 0x80000000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fneg_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = fneg float %v0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fabs_value_f32:
; VI:  v_mul_f32_e64 v{{[0-9]+}}, 1.0, |v{{[0-9]+}}|
; GFX9: v_max_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, |v{{[0-9]+}}|
define amdgpu_kernel void @test_no_fold_canonicalize_fabs_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.fabs.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_no_fold_canonicalize_fcopysign_value_f32:
; VI:  v_mul_f32_e64 v{{[0-9]+}}, 1.0, |v{{[0-9]+}}|
; GFX9: v_max_f32_e64 v{{[0-9]+}}, |v{{[0-9]+}}|, |v{{[0-9]+}}|

; GCN-NOT: v_mul_
; GCN-NOT: v_max_
define amdgpu_kernel void @test_no_fold_canonicalize_fcopysign_value_f32(ptr addrspace(1) %arg, float %sign) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %canon.load = tail call float @llvm.canonicalize.f32(float %load)
  %copysign = call float @llvm.copysign.f32(float %canon.load, float %sign)
  %v = tail call float @llvm.fabs.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fabs_value_f32:
; GCN: v_and_b32_e32 [[V:v[0-9]+]], 0x7fffffff, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_fabs_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.fabs.f32(float %v0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_sin_value_f32:
; GCN: v_sin_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_sin_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.sin.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_cos_value_f32:
; GCN: v_cos_f32_e32 [[V:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_cos_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.cos.f32(float %load)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_sin_value_f16:
; GCN: v_sin_f16_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_short v{{.+}}, [[V0]]
define amdgpu_kernel void @test_fold_canonicalize_sin_value_f16(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, ptr addrspace(1) %arg, i32 %id
  %load = load half, ptr addrspace(1) %gep, align 2
  %v = tail call half @llvm.sin.f16(half %load)
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  store half %canonicalized, ptr addrspace(1) %gep, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_cos_value_f16:
; GCN: v_cos_f16_e32 [[V0:v[0-9]+]], v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_short v{{.+}}, [[V0]]
define amdgpu_kernel void @test_fold_canonicalize_cos_value_f16(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, ptr addrspace(1) %arg, i32 %id
  %load = load half, ptr addrspace(1) %gep, align 2
  %v = tail call half @llvm.cos.f16(half %load)
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  store half %canonicalized, ptr addrspace(1) %gep, align 2
  ret void
}

; GCN-LABEL: test_fold_canonicalize_qNaN_value_f32:
; GCN: v_mov_b32_e32 [[V:v[0-9]+]], 0x7fc00000
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_qNaN_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %canonicalized = tail call float @llvm.canonicalize.f32(float 0x7FF8000000000000)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_minnum_value_from_load_f32_ieee_mode:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]
; VI: v_mul_f32_e32 [[QUIET:v[0-9]+]], 1.0, [[VAL]]
; GFX9: v_max_f32_e32 [[QUIET:v[0-9]+]], [[VAL]], [[VAL]]

; GCN: v_min_f32_e32 [[V:v[0-9]+]], 0, [[QUIET]]
; GCN-NOT: v_max
; GCN-NOT: v_mul

; GFX9: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_minnum_value_from_load_f32_ieee_mode(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.minnum.f32(float %load, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; define amdgpu_kernel void @test_fold_canonicalize_minnum_value_from_load_f32_nnan_ieee_mode(ptr addrspace(1) %arg) #1 {
;   %id = tail call i32 @llvm.amdgcn.workitem.id.x()
;   %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
;   %load = load float, ptr addrspace(1) %gep, align 4
;   %v = tail call float @llvm.minnum.f32(float %load, float 0.0)
;   %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
;   store float %canonicalized, ptr addrspace(1) %gep, align 4
;   ret void
; }

; GCN-LABEL: test_fold_canonicalize_minnum_value_f32:
; GCN: v_min_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_minnum_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.minnum.f32(float %v0, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; FIXME: Should there be more checks here? minnum with NaN operand is simplified away.

; GCN-LABEL: test_fold_canonicalize_sNaN_value_f32:
; GCN: {{flat|global}}_load_dword [[LOAD:v[0-9]+]]
; VI: v_mul_f32_e32 v{{[0-9]+}}, 1.0, [[LOAD]]
; GFX9: v_max_f32_e32 v{{[0-9]+}}, [[LOAD]], [[LOAD]]
define amdgpu_kernel void @test_fold_canonicalize_sNaN_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.minnum.f32(float %load, float bitcast (i32 2139095041 to float))
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; define amdgpu_kernel void @test_fold_canonicalize_denorm_value_f32(ptr addrspace(1) %arg) {
;   %id = tail call i32 @llvm.amdgcn.workitem.id.x()
;   %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
;   %load = load float, ptr addrspace(1) %gep, align 4
;   %v = tail call float @llvm.minnum.f32(float %load, float bitcast (i32 8388607 to float))
;   %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
;   store float %canonicalized, ptr addrspace(1) %gep, align 4
;   ret void
; }

; GCN-LABEL: test_fold_canonicalize_maxnum_value_from_load_f32_ieee_mode:
; GCN: {{flat|global}}_load_dword [[VAL:v[0-9]+]]

; GFX9:  v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[VAL]]

; VI-FLUSH:    v_mul_f32_e32 [[QUIET:v[0-9]+]], 1.0, [[VAL]]
; VI-FLUSH:    v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[QUIET]]

; VI-DENORM: v_max_f32_e32 [[RESULT:v[0-9]+]], 0, [[VAL]]

; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN:  {{flat|global}}_store_dword v{{.+}}, [[RESULT]]
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_from_load_f32_ieee_mode(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v = tail call float @llvm.maxnum.f32(float %load, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_maxnum_value_f32:
; GCN: v_max_f32_e32 [[V:v[0-9]+]], 0, v{{[0-9]+}}
; GCN-NOT: v_max
; GCN-NOT: v_mul
; GCN: {{flat|global}}_store_dword v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load = load float, ptr addrspace(1) %gep, align 4
  %v0 = fadd float %load, 0.0
  %v = tail call float @llvm.maxnum.f32(float %v0, float 0.0)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; GCN-LABEL: test_fold_canonicalize_maxnum_value_f64:
; GCN: v_max_f64 [[V:v\[[0-9]+:[0-9]+\]]], v[{{[0-9:]+}}], 0
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN:  {{flat|global}}_store_dwordx2 v{{.+}}, [[V]]
define amdgpu_kernel void @test_fold_canonicalize_maxnum_value_f64(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, ptr addrspace(1) %arg, i32 %id
  %load = load double, ptr addrspace(1) %gep, align 8
  %v0 = fadd double %load, 0.0
  %v = tail call double @llvm.maxnum.f64(double %v0, double 0.0)
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  store double %canonicalized, ptr addrspace(1) %gep, align 8
  ret void
}

; GCN-LABEL: test_fold_canonicalize_fmul_value_f32_no_ieee:
; GCN: v_mul_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN-NEXT: ; return
define amdgpu_ps float @test_fold_canonicalize_fmul_value_f32_no_ieee(float %arg) {
entry:
  %v = fmul float %arg, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: test_fold_canonicalize_fmul_nnan_value_f32_no_ieee:
; GCN: v_mul_f32_e32 [[V:v[0-9]+]], 0x41700000, v{{[0-9]+}}
; GCN-NOT: v_mul
; GCN-NOT: v_max
; GCN-NEXT: ; return
define amdgpu_ps float @test_fold_canonicalize_fmul_nnan_value_f32_no_ieee(float %arg) {
entry:
  %v = fmul nnan float %arg, 15.0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}test_fold_canonicalize_fdiv_value_f32_no_ieee:
; GCN: v_div_fixup_f32
; GCN-NOT: v_max
; GCN-NOT: v_mul
; GCN: ; return
define amdgpu_ps float @test_fold_canonicalize_fdiv_value_f32_no_ieee(float %arg0) {
entry:
  %v = fdiv float 15.0, %arg0
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f32
; GFX9-DENORM: global_load_dword [[V:v[0-9]+]],
; GFX9-DENORM: global_store_dword v{{[0-9]+}}, [[V]], s{{\[[0-9]+:[0-9]+\]}}
; GFX9-DENORM-NOT: 1.0
; GFX9-DENORM-NOT: v_max
; VI-FLUSH: v_mul_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; GFX9-FLUSH: v_max_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f32(ptr addrspace(1) %arg, ptr addrspace(1) %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %v = load float, ptr addrspace(1) %gep, align 4
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  %gep2 = getelementptr inbounds float, ptr addrspace(1) %out, i32 %id
  store float %canonicalized, ptr addrspace(1) %gep2, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f64
; GCN: {{flat|global}}_load_dwordx2 [[V:v\[[0-9:]+\]]],
; GCN: {{flat|global}}_store_dwordx2 v{{.+}}, [[V]]
; GCN-NOT: v_mul_
; GCN-NOT: v_max_
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f64(ptr addrspace(1) %arg, ptr addrspace(1) %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds double, ptr addrspace(1) %arg, i32 %id
  %v = load double, ptr addrspace(1) %gep, align 8
  %canonicalized = tail call double @llvm.canonicalize.f64(double %v)
  %gep2 = getelementptr inbounds double, ptr addrspace(1) %out, i32 %id
  store double %canonicalized, ptr addrspace(1) %gep2, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_load_nnan_value_f16
; GCN: {{flat|global}}_load_ushort [[V1:v[0-9]+]],
; GCN: v_max_f16_e32 [[V2:v[0-9]+]], [[V1]], [[V1]]
; GCN: {{flat|global}}_store_short v{{.+}}, [[V2]]
define amdgpu_kernel void @test_fold_canonicalize_load_nnan_value_f16(ptr addrspace(1) %arg, ptr addrspace(1) %out) #1 {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds half, ptr addrspace(1) %arg, i32 %id
  %v = load half, ptr addrspace(1) %gep, align 2
  %canonicalized = tail call half @llvm.canonicalize.f16(half %v)
  %gep2 = getelementptr inbounds half, ptr addrspace(1) %out, i32 %id
  store half %canonicalized, ptr addrspace(1) %gep2, align 2
  ret void
}

; GCN-LABEL: {{^}}test_fold_canonicalize_select_value_f32:
; GCN: v_add_f32
; GCN: v_add_f32
; GCN: v_cndmask_b32
; GCN-NOT: v_mul_
; GCN-NOT: v_max_
define amdgpu_kernel void @test_fold_canonicalize_select_value_f32(ptr addrspace(1) %arg) {
  %id = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds float, ptr addrspace(1) %arg, i32 %id
  %load0 = load volatile float, ptr addrspace(1) %gep, align 4
  %load1 = load volatile float, ptr addrspace(1) %gep, align 4
  %load2 = load volatile i32, ptr addrspace(1) undef, align 4
  %v0 = fadd float %load0, 15.0
  %v1 = fadd float %load1, 32.0
  %cond = icmp eq i32 %load2, 0
  %select = select i1 %cond, float %v0, float %v1
  %canonicalized = tail call float @llvm.canonicalize.f32(float %select)
  store float %canonicalized, ptr addrspace(1) %gep, align 4
  ret void
}

; Need to quiet the nan with a separate instruction since it will be
; passed through the minnum.
; FIXME: canonicalize doens't work correctly without ieee_mode

; GCN-LABEL: {{^}}test_fold_canonicalize_minnum_value_no_ieee_mode:
; GFX9-NOT: v0
; GFX9-NOT: v1
; GFX9: v_min_f32_e32 v0, v0, v1
; GFX9-NEXT: ; return to shader

; VI-FLUSH: v_min_f32_e32 v0, v0, v1
; VI-FLUSH-NEXT: v_mul_f32_e32 v0, 1.0, v0
; VI-FLUSH-NEXT: ; return

; VI-DENORM-NOT: v0
; VI-DENORM: v_min_f32_e32 v0, v0, v1
; VI-DENORM-NEXT: ; return
define amdgpu_ps float @test_fold_canonicalize_minnum_value_no_ieee_mode(float %arg0, float %arg1) {
  %v = tail call float @llvm.minnum.f32(float %arg0, float %arg1)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}test_fold_canonicalize_minnum_value_ieee_mode:
; GFX9: v_min_f32_e32 v0, v0, v1
; GFX9-NEXT: s_setpc_b64

; VI-DAG: v_mul_f32_e32 v0, 1.0, v0
; VI-DAG: v_mul_f32_e32 v1, 1.0, v1
; VI: v_min_f32_e32 v0, v0, v1

; VI-NEXT: s_setpc_b64
define float @test_fold_canonicalize_minnum_value_ieee_mode(float %arg0, float %arg1) {
  %v = tail call float @llvm.minnum.f32(float %arg0, float %arg1)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; Canonicalizing flush necessary pre-gfx9
; GCN-LABEL: {{^}}test_fold_canonicalize_minnum_value_no_ieee_mode_nnan:
; GCN: v_min_f32_e32 v0, v0, v1
; VI-FLUSH-NEXT: v_mul_f32_e32 v0, 1.0, v0
; GCN-NEXT: ; return
define amdgpu_ps float @test_fold_canonicalize_minnum_value_no_ieee_mode_nnan(float %arg0, float %arg1) #1 {
  %v = tail call float @llvm.minnum.f32(float %arg0, float %arg1)
  %canonicalized = tail call float @llvm.canonicalize.f32(float %v)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_build_vector_v2f16:
; GFX9-DAG: v_add_f16_e32
; GFX9-DAG: v_mul_f16_e32
; GFX9-NOT: v_max
; GFX9-NOT: v_pk_max
define <2 x half> @v_test_canonicalize_build_vector_v2f16(<2 x half> %vec) {
  %lo = extractelement <2 x half> %vec, i32 0
  %hi = extractelement <2 x half> %vec, i32 1
  %lo.op = fadd half %lo, 1.0
  %hi.op = fmul half %lo, 4.0
  %ins0 = insertelement <2 x half> undef, half %lo.op, i32 0
  %ins1 = insertelement <2 x half> %ins0, half %hi.op, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins1)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_build_vector_noncanon1_v2f16:
; GFX9: v_add_f16_e32
; GFX9: v_pk_max
define <2 x half> @v_test_canonicalize_build_vector_noncanon1_v2f16(<2 x half> %vec) {
  %lo = extractelement <2 x half> %vec, i32 0
  %lo.op = fadd half %lo, 1.0
  %ins = insertelement <2 x half> %vec, half %lo.op, i32 0
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_build_vector_noncanon0_v2f16:
; GFX9: v_add_f16_sdwa
; GFX9: v_pk_max
define <2 x half> @v_test_canonicalize_build_vector_noncanon0_v2f16(<2 x half> %vec) {
  %hi = extractelement <2 x half> %vec, i32 1
  %hi.op = fadd half %hi, 1.0
  %ins = insertelement <2 x half> %vec, half %hi.op, i32 1
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_extract_element_v2f16:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mul_f16_e32 v0, 4.0, v0
; GFX9-NEXT: s_setpc_b64
define half @v_test_canonicalize_extract_element_v2f16(<2 x half> %vec) {
  %vec.op = fmul <2 x half> %vec, <half 4.0, half 4.0>
  %elt = extractelement <2 x half> %vec.op, i32 0
  %canonicalized = call half @llvm.canonicalize.f16(half %elt)
  ret half %canonicalized
}

; define <2 x half> @v_test_canonicalize_insertelement_v2f16(<2 x half> %vec, half %val, i32 %idx) {
;   %vec.op = fmul <2 x half> %vec, <half 4.0, half 4.0>
;   %ins.op = fmul half %val, 8.0
;   %ins = insertelement <2 x half> %vec.op, half %ins.op, i32 %idx
;   %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins)
;   ret <2 x half> %canonicalized
; }

; GCN-LABEL: {{^}}v_test_canonicalize_insertelement_noncanon_vec_v2f16:
; GFX9: v_mul_f16
; GFX9: v_pk_max_f16 v0, v0, v0
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_insertelement_noncanon_vec_v2f16(<2 x half> %vec, half %val, i32 %idx) {
  %ins.op = fmul half %val, 8.0
  %ins = insertelement <2 x half> %vec, half %ins.op, i32 %idx
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins)
  ret <2 x half> %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_insertelement_noncanon_insval_v2f16:
; GFX9: v_pk_mul_f16
; GFX9: v_pk_max_f16 v0, v0, v0
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_test_canonicalize_insertelement_noncanon_insval_v2f16(<2 x half> %vec, half %val, i32 %idx) {
  %vec.op = fmul <2 x half> %vec, <half 4.0, half 4.0>
  %ins = insertelement <2 x half> %vec.op, half %val, i32 %idx
  %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %ins)
  ret <2 x half> %canonicalized
}

; define <2 x half> @v_test_canonicalize_cvt_pkrtz(float %a, float %b) {
;   %cvt = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %a, float %b)
;   %canonicalized = call <2 x half> @llvm.canonicalize.v2f16(<2 x half> %cvt)
;   ret <2 x half> %canonicalized
; }

; GCN-LABEL: {{^}}v_test_canonicalize_cubeid:
; GCN: s_waitcnt
; GCN-NEXT: v_cubeid_f32 v0, v0, v1, v2
; GCN-NEXT: s_setpc_b64
define float @v_test_canonicalize_cubeid(float %a, float %b, float %c) {
  %cvt = call float @llvm.amdgcn.cubeid(float %a, float %b, float %c)
  %canonicalized = call float @llvm.canonicalize.f32(float %cvt)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_frexp_mant:
; GCN: s_waitcnt
; GCN-NEXT: v_frexp_mant_f32_e32 v0, v0
; GCN-NEXT: s_setpc_b64
define float @v_test_canonicalize_frexp_mant(float %a) {
  %cvt = call float @llvm.amdgcn.frexp.mant.f32(float %a)
  %canonicalized = call float @llvm.canonicalize.f32(float %cvt)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_amdgcn_log:
; GCN: s_waitcnt
; GCN-NEXT: v_log_f32
; GCN-NEXT: s_setpc_b64
define float @v_test_canonicalize_amdgcn_log(float %a) {
  %log = call float @llvm.amdgcn.log.f32(float %a)
  %canonicalized = call float @llvm.canonicalize.f32(float %log)
  ret float %canonicalized
}

; GCN-LABEL: {{^}}v_test_canonicalize_amdgcn_exp2:
; GCN: s_waitcnt
; GCN-NEXT: v_exp_f32
; GCN-NEXT: s_setpc_b64
define float @v_test_canonicalize_amdgcn_exp2(float %a) {
  %log = call float @llvm.amdgcn.exp2.f32(float %a)
  %canonicalized = call float @llvm.canonicalize.f32(float %log)
  ret float %canonicalized
}

; Avoid failing the test on FreeBSD11.0 which will match the GCN-NOT: 1.0
; in the .amd_amdgpu_isa "amdgcn-unknown-freebsd11.0--gfx802" directive
; GCN: .amd_amdgpu_isa

declare float @llvm.canonicalize.f32(float) #0
declare float @llvm.copysign.f32(float, float) #0
declare float @llvm.amdgcn.fmul.legacy(float, float) #0
declare float @llvm.amdgcn.fmad.ftz.f32(float, float, float) #0
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
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #0
declare float @llvm.amdgcn.cubeid(float, float, float) #0
declare float @llvm.amdgcn.frexp.mant.f32(float) #0
declare float @llvm.amdgcn.log.f32(float) #0
declare float @llvm.amdgcn.exp2.f32(float) #0

attributes #0 = { nounwind readnone }
attributes #1 = { "no-nans-fp-math"="true" }
attributes #2 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" }
