; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=NOSNAN -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mattr=+fp-exceptions -verify-machineinstrs < %s | FileCheck -check-prefix=SNAN -check-prefix=GCN %s

declare i32 @llvm.r600.read.tidig.x() #0
declare float @llvm.minnum.f32(float, float) #0
declare float @llvm.maxnum.f32(float, float) #0
declare double @llvm.minnum.f64(double, double) #0
declare double @llvm.maxnum.f64(double, double) #0

; GCN-LABEL: {{^}}v_test_fmed3_r_i_i_f32:
; NOSNAN: v_med3_f32 v{{[0-9]+}}, v{{[0-9]+}}, 2.0, 4.0

; SNAN: v_max_f32_e32 v{{[0-9]+}}, 2.0, v{{[0-9]+}}
; SNAN: v_min_f32_e32 v{{[0-9]+}}, 4.0, v{{[0-9]+}}
define void @v_test_fmed3_r_i_i_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_commute0_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_commute1_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_constant_order_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_multi_use_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_f64(double addrspace(1)* %out, double addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_fmed3_r_i_i_no_nans_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #2 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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
define void @v_test_legacy_fmed3_r_i_i_f32(float addrspace(1)* %out, float addrspace(1)* %aptr) #1 {
  %tid = call i32 @llvm.r600.read.tidig.x()
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

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" "no-nans-fp-math"="false" }
attributes #2 = { nounwind "unsafe-fp-math"="false" "no-nans-fp-math"="true" }
