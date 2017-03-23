; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=NOSNAN -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mattr=+fp-exceptions -verify-machineinstrs < %s | FileCheck -check-prefix=SNAN -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=NOSNAN -check-prefix=GCN -check-prefix=VI -check-prefix=GFX89 %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp-exceptions -verify-machineinstrs < %s | FileCheck -check-prefix=SNAN -check-prefix=GCN -check-prefix=VI -check-prefix=GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -verify-machineinstrs < %s | FileCheck -check-prefix=NOSNAN -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx901 -mattr=+fp-exceptions -verify-machineinstrs < %s | FileCheck -check-prefix=SNAN -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX89 %s


; GCN-LABEL: {{^}}v_test_nnan_input_fmed3_r_i_i_f32:
; GCN: v_add_f32_e32 [[ADD:v[0-9]+]], 1.0, v{{[0-9]+}}
; GCN: v_med3_f32 v{{[0-9]+}}, [[ADD]], 2.0, 4.0
define amdgpu_kernel void @v_test_nnan_input_fmed3_r_i_i_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0
  %a.add = fadd nnan float %a, 1.0
  %max = call float @llvm.maxnum.f32(float %a.add, float 2.0)
  %med = call float @llvm.minnum.f32(float %max, float 4.0)

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_f32:
; NOSNAN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0

; SNAN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; SNAN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_fmed3_r_i_i_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float %a, float 2.0)
  %med = call float @llvm.minnum.f32(float %max, float 4.0)

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_commute0_f32:
; NOSNAN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0

; SNAN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; SNAN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_fmed3_r_i_i_commute0_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float 2.0, float %a)
  %med = call float @llvm.minnum.f32(float 4.0, float %max)

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_commute1_f32:
; NOSNAN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0

; SNAN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; SNAN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_fmed3_r_i_i_commute1_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float %a, float 2.0)
  %med = call float @llvm.minnum.f32(float 4.0, float %max)

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_constant_order_f32:
; GCN: v_max_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
; GCN: v_min_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_fmed3_r_i_i_constant_order_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float %a, float 4.0)
  %med = call float @llvm.minnum.f32(float %max, float 2.0)

  store float %med, float addrspace(1)* %outgep
  ret void
}


; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_multi_use_f32:
; GCN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; GCN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_fmed3_r_i_i_multi_use_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float %a, float 2.0)
  %med = call float @llvm.minnum.f32(float %max, float 4.0)

  store volatile float %med, float addrspace(1)* %outgep
  store volatile float %max, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_f64:
; GCN: v_max_f64 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, 2.0
; GCN: v_min_f64 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, 4.0
define amdgpu_kernel void @v_test_fmed3_r_i_i_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr double, double addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr double, double addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %gep0

  %max = call double @llvm.maxnum.f64(double %a, double 2.0)
  %med = call double @llvm.minnum.f64(double %max, double 4.0)

  store double %med, double addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_no_nans_f32:
; GCN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0
define amdgpu_kernel void @v_test_fmed3_r_i_i_no_nans_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  %max = call float @llvm.maxnum.f32(float %a, float 2.0)
  %med = call float @llvm.minnum.f32(float %max, float 4.0)

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_legacy_fmed3_r_i_i_f32:
; NOSNAN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0

; SNAN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; SNAN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define amdgpu_kernel void @v_test_legacy_fmed3_r_i_i_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep0

  ; fmax_legacy
  %cmp0 = fcmp ule float %a, 2.0
  %max = select i1 %cmp0, float 2.0, float %a

  ; fmin_legacy
  %cmp1 = fcmp uge float %max, 4.0
  %med = select i1 %cmp1, float 4.0, float %max

  store float %med, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_srcmod0:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, -[[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_srcmod0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %a.fneg = fsub float -0.0, %a
  %tmp0 = call float @llvm.minnum.f32(float %a.fneg, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a.fneg, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_srcmod1:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], -[[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_srcmod1(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %b.fneg = fsub float -0.0, %b
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b.fneg)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b.fneg)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_srcmod2:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], -[[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_srcmod2(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %c.fneg = fsub float -0.0, %c
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.fneg)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_srcmod012:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, -[[A]], |[[B]]|, -|[[C]]|
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_srcmod012(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.fneg = fsub float -0.0, %a
  %b.fabs = call float @llvm.fabs.f32(float %b)
  %c.fabs = call float @llvm.fabs.f32(float %c)
  %c.fabs.fneg = fsub float -0.0, %c.fabs

  %tmp0 = call float @llvm.minnum.f32(float %a.fneg, float %b.fabs)
  %tmp1 = call float @llvm.maxnum.f32(float %a.fneg, float %b.fabs)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.fabs.fneg)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)

  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_negabs012:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, -|[[A]]|, -|[[B]]|, -|[[C]]|
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_negabs012(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.fabs = call float @llvm.fabs.f32(float %a)
  %a.fabs.fneg = fsub float -0.0, %a.fabs
  %b.fabs = call float @llvm.fabs.f32(float %b)
  %b.fabs.fneg = fsub float -0.0, %b.fabs
  %c.fabs = call float @llvm.fabs.f32(float %c)
  %c.fabs.fneg = fsub float -0.0, %c.fabs

  %tmp0 = call float @llvm.minnum.f32(float %a.fabs.fneg, float %b.fabs.fneg)
  %tmp1 = call float @llvm.maxnum.f32(float %a.fabs.fneg, float %b.fabs.fneg)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.fabs.fneg)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)

  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_nnan_inputs_med3_f32_pat0:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[A_ADD:v[0-9]+]], 1.0, [[A]]
; GCN-DAG: v_add_f32_e32 [[B_ADD:v[0-9]+]], 2.0, [[B]]
; GCN-DAG: v_add_f32_e32 [[C_ADD:v[0-9]+]], 4.0, [[C]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A_ADD]], [[B_ADD]], [[C_ADD]]
define amdgpu_kernel void @v_nnan_inputs_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.nnan = fadd nnan float %a, 1.0
  %b.nnan = fadd nnan float %b, 2.0
  %c.nnan = fadd nnan float %c, 4.0

  %tmp0 = call float @llvm.minnum.f32(float %a.nnan, float %b.nnan)
  %tmp1 = call float @llvm.maxnum.f32(float %a.nnan, float %b.nnan)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.nnan)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; 16 combinations

; 0: max(min(x, y), min(max(x, y), z))
; 1: max(min(x, y), min(max(y, x), z))
; 2: max(min(x, y), min(z, max(x, y)))
; 3: max(min(x, y), min(z, max(y, x)))
; 4: max(min(y, x), min(max(x, y), z))
; 5: max(min(y, x), min(max(y, x), z))
; 6: max(min(y, x), min(z, max(x, y)))
; 7: max(min(y, x), min(z, max(y, x)))
;
; + commute outermost max

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat1:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat1(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat2:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat2(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat3:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat3(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat4:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat4(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat5:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat5(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat6:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat6(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat7:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat7(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat8:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat8(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat9:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat9(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat10:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat10(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat11:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat11(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat12:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat12(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat13:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat13(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat14:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[A]], [[B]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat14(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat15:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_med3_f32 v{{[0-9]+}}, [[B]], [[A]], [[C]]
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat15(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %b, float %a)
  %tmp1 = call float @llvm.maxnum.f32(float %b, float %a)
  %tmp2 = call float @llvm.minnum.f32(float %c, float %tmp1)
  %med3 = call float @llvm.maxnum.f32(float %tmp2, float %tmp0)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; ---------------------------------------------------------------------
; Negative patterns
; ---------------------------------------------------------------------

; GCN-LABEL: {{^}}v_test_safe_med3_f32_pat0_multi_use0:
; GCN-DAG: v_min_f32
; GCN-DAG: v_max_f32
; GCN: v_min_f32
; GCN: v_max_f32
define amdgpu_kernel void @v_test_safe_med3_f32_pat0_multi_use0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  store volatile float %tmp0, float addrspace(1)* undef
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_safe_med3_f32_pat0_multi_use1:
define amdgpu_kernel void @v_test_safe_med3_f32_pat0_multi_use1(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  store volatile float %tmp1, float addrspace(1)* undef
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_safe_med3_f32_pat0_multi_use2:
define amdgpu_kernel void @v_test_safe_med3_f32_pat0_multi_use2(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  store volatile float %tmp2, float addrspace(1)* undef
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}


; GCN-LABEL: {{^}}v_test_safe_med3_f32_pat0:
define amdgpu_kernel void @v_test_safe_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %tmp0 = call float @llvm.minnum.f32(float %a, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_nnan_inputs_missing0_med3_f32_pat0:
define amdgpu_kernel void @v_nnan_inputs_missing0_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.nnan = fadd float %a, 1.0
  %b.nnan = fadd nnan float %b, 2.0
  %c.nnan = fadd nnan float %c, 4.0

  %tmp0 = call float @llvm.minnum.f32(float %a.nnan, float %b.nnan)
  %tmp1 = call float @llvm.maxnum.f32(float %a.nnan, float %b.nnan)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.nnan)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_nnan_inputs_missing1_med3_f32_pat0:
define amdgpu_kernel void @v_nnan_inputs_missing1_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.nnan = fadd nnan float %a, 1.0
  %b.nnan = fadd float %b, 2.0
  %c.nnan = fadd nnan float %c, 4.0

  %tmp0 = call float @llvm.minnum.f32(float %a.nnan, float %b.nnan)
  %tmp1 = call float @llvm.maxnum.f32(float %a.nnan, float %b.nnan)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.nnan)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_nnan_inputs_missing2_med3_f32_pat0:
define amdgpu_kernel void @v_nnan_inputs_missing2_med3_f32_pat0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2

  %a.nnan = fadd nnan float %a, 1.0
  %b.nnan = fadd nnan float %b, 2.0
  %c.nnan = fadd float %c, 4.0

  %tmp0 = call float @llvm.minnum.f32(float %a.nnan, float %b.nnan)
  %tmp1 = call float @llvm.maxnum.f32(float %a.nnan, float %b.nnan)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c.nnan)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_global_nnans_med3_f32_pat0_srcmod0_mismatch:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_min_f32
; GCN: v_max_f32
; GCN: v_min_f32
; GCN: v_max_f32
define amdgpu_kernel void @v_test_global_nnans_med3_f32_pat0_srcmod0_mismatch(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %a.fneg = fsub float -0.0, %a
  %tmp0 = call float @llvm.minnum.f32(float %a.fneg, float %b)
  %tmp1 = call float @llvm.maxnum.f32(float %a, float %b)
  %tmp2 = call float @llvm.minnum.f32(float %tmp1, float %c)
  %med3 = call float @llvm.maxnum.f32(float %tmp0, float %tmp2)
  store float %med3, float addrspace(1)* %outgep
  ret void
}

; A simple min and max is not sufficient
; GCN-LABEL: {{^}}v_test_global_nnans_min_max_f32:
; GCN: {{buffer_|flat_}}load_dword [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_dword [[C:v[0-9]+]]
; GCN: v_max_f32_e32 [[MAX:v[0-9]+]], [[B]], [[A]]
; GCN: v_min_f32_e32 v{{[0-9]+}}, [[C]], [[MAX]]
define amdgpu_kernel void @v_test_global_nnans_min_max_f32(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) #2 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr float, float addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr float, float addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr float, float addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %a = load volatile float, float addrspace(1)* %gep0
  %b = load volatile float, float addrspace(1)* %gep1
  %c = load volatile float, float addrspace(1)* %gep2
  %max = call float @llvm.maxnum.f32(float %a, float %b)
  %minmax = call float @llvm.minnum.f32(float %max, float %c)
  store float %minmax, float addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_test_nnan_input_fmed3_r_i_i_f16:
; SI: v_cvt_f32_f16
; SI: v_add_f32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}
; SI: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0
; SI: v_cvt_f16_f32

; VI: v_add_f16_e32 v{{[0-9]+}}, 1.0
; VI: v_max_f16_e32 v{{[0-9]+}}, 2.0
; VI: v_min_f16_e32 v{{[0-9]+}}, 4.0

; GFX9: v_add_f16_e32 v{{[0-9]+}}, 1.0
; GFX9: v_med3_f16 v{{[0-9]+}}, [[ADD]], 2.0, 4.0
define amdgpu_kernel void @v_test_nnan_input_fmed3_r_i_i_f16(half addrspace(1)* %out, half addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %outgep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load half, half addrspace(1)* %gep0
  %a.add = fadd nnan half %a, 1.0
  %max = call half @llvm.maxnum.f16(half %a.add, half 2.0)
  %med = call half @llvm.minnum.f16(half %max, half 4.0)

  store half %med, half addrspace(1)* %outgep
  ret void
}

; GCN-LABEL: {{^}}v_nnan_inputs_med3_f16_pat0:
; GCN: {{buffer_|flat_}}load_ushort [[A:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_ushort [[B:v[0-9]+]]
; GCN: {{buffer_|flat_}}load_ushort [[C:v[0-9]+]]

; SI: v_cvt_f32_f16
; SI: v_cvt_f32_f16
; SI: v_add_f32_e32
; SI: v_add_f32_e32
; SI: v_add_f32_e32
; SI: v_med3_f32
; SI: v_cvt_f16_f32_e32


; GFX89-DAG: v_add_f16_e32 [[A_ADD:v[0-9]+]], 1.0, [[A]]
; GFX89-DAG: v_add_f16_e32 [[B_ADD:v[0-9]+]], 2.0, [[B]]
; GFX89-DAG: v_add_f16_e32 [[C_ADD:v[0-9]+]], 4.0, [[C]]

; VI-DAG: v_min_f16
; VI-DAG: v_max_f16
; VI: v_min_f16
; VI: v_max_f16

; GFX9: v_med3_f16 v{{[0-9]+}}, [[A_ADD]], [[B_ADD]], [[C_ADD]]
define amdgpu_kernel void @v_nnan_inputs_med3_f16_pat0(half addrspace(1)* %out, half addrspace(1)* %aptr, half addrspace(1)* %bptr, half addrspace(1)* %cptr) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep0 = getelementptr half, half addrspace(1)* %aptr, i32 %tid
  %gep1 = getelementptr half, half addrspace(1)* %bptr, i32 %tid
  %gep2 = getelementptr half, half addrspace(1)* %cptr, i32 %tid
  %outgep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %a = load volatile half, half addrspace(1)* %gep0
  %b = load volatile half, half addrspace(1)* %gep1
  %c = load volatile half, half addrspace(1)* %gep2

  %a.nnan = fadd nnan half %a, 1.0
  %b.nnan = fadd nnan half %b, 2.0
  %c.nnan = fadd nnan half %c, 4.0

  %tmp0 = call half @llvm.minnum.f16(half %a.nnan, half %b.nnan)
  %tmp1 = call half @llvm.maxnum.f16(half %a.nnan, half %b.nnan)
  %tmp2 = call half @llvm.minnum.f16(half %tmp1, half %c.nnan)
  %med3 = call half @llvm.maxnum.f16(half %tmp0, half %tmp2)
  store half %med3, half addrspace(1)* %outgep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare float @llvm.fabs.f32(float) #0
declare float @llvm.minnum.f32(float, float) #0
declare float @llvm.maxnum.f32(float, float) #0
declare double @llvm.minnum.f64(double, double) #0
declare double @llvm.maxnum.f64(double, double) #0
declare half @llvm.fabs.f16(half) #0
declare half @llvm.minnum.f16(half, half) #0
declare half @llvm.maxnum.f16(half, half) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" "no-nans-fp-math"="false" }
attributes #2 = { nounwind "unsafe-fp-math"="false" "no-nans-fp-math"="true" }
