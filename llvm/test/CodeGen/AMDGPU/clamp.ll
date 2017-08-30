; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX89,GFX9 %s

; GCN-LABEL: {{^}}v_clamp_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neg_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, -[[A]], -[[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_neg_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %fneg.a = fsub float -0.0, %a
  %max = call float @llvm.maxnum.f32(float %fneg.a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_negabs_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, -|[[A]]|, -|[[A]]| clamp{{$}}
define amdgpu_kernel void @v_clamp_negabs_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %fabs.a = call float @llvm.fabs.f32(float %a)
  %fneg.fabs.a = fsub float -0.0, %fabs.a

  %max = call float @llvm.maxnum.f32(float %fneg.fabs.a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_negzero_f32:
; GCN-DAG: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN-DAG: v_bfrev_b32_e32 [[SIGNBIT:v[0-9]+]], 1
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[SIGNBIT]], 1.0
define amdgpu_kernel void @v_clamp_negzero_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float -0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_multi_use_max_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], 0, [[A]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], 1.0, [[MAX]]
define amdgpu_kernel void @v_clamp_multi_use_max_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  store volatile float %max, float addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}v_clamp_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX89: v_max_f16_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}

; SI: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], [[A]] clamp{{$}}
; SI: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[CVT]]
define amdgpu_kernel void @v_clamp_f16(half addrspace(1)* %out, half addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %max = call half @llvm.maxnum.f16(half %a, half 0.0)
  %med = call half @llvm.minnum.f16(half %max, half 1.0)

  store half %med, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neg_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX89: v_max_f16_e64 v{{[0-9]+}}, -[[A]], -[[A]] clamp{{$}}

; FIXME: Better to fold neg into max
; SI: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -[[A]] clamp{{$}}
; SI: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[CVT]]
define amdgpu_kernel void @v_clamp_neg_f16(half addrspace(1)* %out, half addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %fneg.a = fsub half -0.0, %a
  %max = call half @llvm.maxnum.f16(half %fneg.a, half 0.0)
  %med = call half @llvm.minnum.f16(half %max, half 1.0)

  store half %med, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_negabs_f16:
; GCN: {{buffer|flat|global}}_load_ushort [[A:v[0-9]+]]
; GFX89: v_max_f16_e64 v{{[0-9]+}}, -|[[A]]|, -|[[A]]| clamp{{$}}

; FIXME: Better to fold neg/abs into max

; SI: v_cvt_f32_f16_e64 [[CVT:v[0-9]+]], -|[[A]]| clamp{{$}}
; SI: v_cvt_f16_f32_e32 v{{[0-9]+}}, [[CVT]]
define amdgpu_kernel void @v_clamp_negabs_f16(half addrspace(1)* %out, half addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %fabs.a = call half @llvm.fabs.f16(half %a)
  %fneg.fabs.a = fsub half -0.0, %fabs.a

  %max = call half @llvm.maxnum.f16(half %fneg.fabs.a, half 0.0)
  %med = call half @llvm.minnum.f16(half %max, half 1.0)

  store half %med, half addrspace(1)* %out.gep
  ret void
}

; FIXME: Do f64 instructions support clamp?
; GCN-LABEL: {{^}}v_clamp_f64:
; GCN: {{buffer|flat|global}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_max_f64 v{{\[[0-9]+:[0-9]+\]}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr double, double addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %gep0
  %max = call double @llvm.maxnum.f64(double %a, double 0.0)
  %med = call double @llvm.minnum.f64(double %max, double 1.0)

  store double %med, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neg_f64:
; GCN: {{buffer|flat|global}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_max_f64 v{{\[[0-9]+:[0-9]+\]}}, -[[A]], -[[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_neg_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr double, double addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %gep0
  %fneg.a = fsub double -0.0, %a
  %max = call double @llvm.maxnum.f64(double %fneg.a, double 0.0)
  %med = call double @llvm.minnum.f64(double %max, double 1.0)

  store double %med, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_negabs_f64:
; GCN: {{buffer|flat|global}}_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; GCN: v_max_f64 v{{\[[0-9]+:[0-9]+\]}}, -|[[A]]|, -|[[A]]| clamp{{$}}
define amdgpu_kernel void @v_clamp_negabs_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr double, double addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %gep0
  %fabs.a = call double @llvm.fabs.f64(double %a)
  %fneg.fabs.a = fsub double -0.0, %fabs.a

  %max = call double @llvm.maxnum.f64(double %fneg.fabs.a, double 0.0)
  %med = call double @llvm.minnum.f64(double %max, double 1.0)

  store double %med, double addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_aby_negzero_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32
define amdgpu_kernel void @v_clamp_med3_aby_negzero_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float -0.0, float 1.0, float %a)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_aby_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_aby_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float %a)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_bay_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_bay_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 1.0, float 0.0, float %a)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_yab_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_yab_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float %a, float 0.0, float 1.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_yba_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_yba_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float %a, float 1.0, float 0.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_ayb_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_ayb_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float %a, float 1.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_bya_f32:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_bya_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 1.0, float %a, float 0.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constants_to_one_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 1.0
define amdgpu_kernel void @v_clamp_constants_to_one_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float 4.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constants_to_zero_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @v_clamp_constants_to_zero_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float -4.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_preserve_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0.5
define amdgpu_kernel void @v_clamp_constant_preserve_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float 0.5)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_preserve_denorm_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0x7fffff{{$}}
define amdgpu_kernel void @v_clamp_constant_preserve_denorm_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float bitcast (i32 8388607 to float))
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_qnan_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @v_clamp_constant_qnan_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float 0x7FF8000000000000)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_snan_f32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @v_clamp_constant_snan_f32(float addrspace(1)* %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float bitcast (i32 2139095041 to float))
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; ---------------------------------------------------------------------
; Test non-default behaviors enabling snans and disabling dx10_clamp
; ---------------------------------------------------------------------

; GCN-LABEL: {{^}}v_clamp_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], 0, 1.0
define amdgpu_kernel void @v_clamp_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_f32_snan_dx10clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_f32_snan_dx10clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #3 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_f32_snan_no_dx10clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], 0, [[A]]
; GCN: v_min_f32_e32 [[MIN:v[0-9]+]], 1.0, [[MAX]]
define amdgpu_kernel void @v_clamp_f32_snan_no_dx10clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #4 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %max = call float @llvm.maxnum.f32(float %a, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_f32_snan_no_dx10clamp_nnan_src:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], 0, 1.0
define amdgpu_kernel void @v_clamp_f32_snan_no_dx10clamp_nnan_src(float addrspace(1)* %out, float addrspace(1)* %aptr) #4 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %add  = fadd nnan float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %add, float 0.0)
  %med = call float @llvm.minnum.f32(float %max, float 1.0)

  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_aby_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_aby_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float %a)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_bay_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_max_f32_e64 v{{[0-9]+}}, [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_med3_bay_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 1.0, float 0.0, float %a)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_yab_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], 0, 1.0
define amdgpu_kernel void @v_clamp_med3_yab_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float %a, float 0.0, float 1.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_yba_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], 1.0, 0
define amdgpu_kernel void @v_clamp_med3_yba_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float %a, float 1.0, float 0.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_ayb_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, 0, [[A]], 1.0
define amdgpu_kernel void @v_clamp_med3_ayb_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float %a, float 1.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_med3_bya_f32_no_dx10_clamp:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, 1.0, [[A]], 0
define amdgpu_kernel void @v_clamp_med3_bya_f32_no_dx10_clamp(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %med = call float @llvm.amdgcn.fmed3.f32(float 1.0, float %a, float 0.0)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_qnan_f32_no_dx10_clamp:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0x7fc00000
define amdgpu_kernel void @v_clamp_constant_qnan_f32_no_dx10_clamp(float addrspace(1)* %out) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float 0x7FF8000000000000)
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_constant_snan_f32_no_dx10_clamp:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0x7f800001
define amdgpu_kernel void @v_clamp_constant_snan_f32_no_dx10_clamp(float addrspace(1)* %out) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %med = call float @llvm.amdgcn.fmed3.f32(float 0.0, float 1.0, float bitcast (i32 2139095041 to float))
  store float %med, float addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_v2f16:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_v2f16_undef_elt:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] clamp{{$}}
define amdgpu_kernel void @v_clamp_v2f16_undef_elt(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> <half undef, half 0.0>)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half undef>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_v2f16_not_zero:
; GFX9: v_pk_max_f16
; GFX9: v_pk_min_f16
define amdgpu_kernel void @v_clamp_v2f16_not_zero(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> <half 2.0, half 0.0>)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_v2f16_not_one:
; GFX9: v_pk_max_f16
; GFX9: v_pk_min_f16
define amdgpu_kernel void @v_clamp_v2f16_not_one(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %a, <2 x half> <half 0.0, half 0.0>)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 0.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neg_v2f16:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] neg_lo:[1,1] neg_hi:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_neg_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %fneg.a = fsub <2 x half> <half -0.0, half -0.0>, %a
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %fneg.a, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_negabs_v2f16:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9: v_and_b32_e32 [[ABS:v[0-9]+]], 0x7fff7fff, [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[ABS]], [[ABS]] neg_lo:[1,1] neg_hi:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_negabs_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %fabs.a = call <2 x half> @llvm.fabs.v2f16(<2 x half> %a)
  %fneg.fabs.a = fsub <2 x half> <half -0.0, half -0.0>, %fabs.a

  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %fneg.fabs.a, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neglo_v2f16:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] neg_lo:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_neglo_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %lo = extractelement <2 x half> %a, i32 0
  %neg.lo = fsub half -0.0, %lo
  %neg.lo.vec = insertelement <2 x half> %a, half %neg.lo, i32 0
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %neg.lo.vec, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_neghi_v2f16:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] neg_hi:[1,1] clamp{{$}}
define amdgpu_kernel void @v_clamp_neghi_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %hi = extractelement <2 x half> %a, i32 1
  %neg.hi = fsub half -0.0, %hi
  %neg.hi.vec = insertelement <2 x half> %a, half %neg.hi, i32 1
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %neg.hi.vec, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_clamp_v2f16_shuffle:
; GCN: {{buffer|flat|global}}_load_dword [[A:v[0-9]+]]
; GFX9-NOT: [[A]]
; GFX9: v_pk_max_f16 [[CLAMP:v[0-9]+]], [[A]], [[A]] op_sel:[1,1] op_sel_hi:[0,0] clamp{{$}}
define amdgpu_kernel void @v_clamp_v2f16_shuffle(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %aptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr <2 x half>, <2 x half> addrspace(1)* %aptr, i32 %tid
  %out.gep = getelementptr <2 x half>, <2 x half> addrspace(1)* %out, i32 %tid
  %a = load <2 x half>, <2 x half> addrspace(1)* %gep0
  %shuf = shufflevector <2 x half> %a, <2 x half> undef, <2 x i32> <i32 1, i32 0>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %shuf, <2 x half> zeroinitializer)
  %med = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)

  store <2 x half> %med, <2 x half> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fabs.f32(float) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare float @llvm.amdgcn.fmed3.f32(float, float, float) #1
declare double @llvm.fabs.f64(double) #1
declare double @llvm.minnum.f64(double, double) #1
declare double @llvm.maxnum.f64(double, double) #1
declare half @llvm.fabs.f16(half) #1
declare half @llvm.minnum.f16(half, half) #1
declare half @llvm.maxnum.f16(half, half) #1
declare <2 x half> @llvm.fabs.v2f16(<2 x half>) #1
declare <2 x half> @llvm.minnum.v2f16(<2 x half>, <2 x half>) #1
declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "target-features"="-dx10-clamp,-fp-exceptions" "no-nans-fp-math"="false" }
attributes #3 = { nounwind "target-features"="+dx10-clamp,+fp-exceptions" "no-nans-fp-math"="false" }
attributes #4 = { nounwind "target-features"="-dx10-clamp,+fp-exceptions" "no-nans-fp-math"="false" }
