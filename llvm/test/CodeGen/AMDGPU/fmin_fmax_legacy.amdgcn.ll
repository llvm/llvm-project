; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SI-SAFE,GCN %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s

; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI-SAFE,GCN %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s

; GCN-LABEL: {{^}}min_fneg_select_regression_0:
; GCN-NOT: v_mul

; SI-SAFE: v_max_legacy_f32_e64 [[MIN:v[0-9]+]], -1.0, -v0

; VI-SAFE: v_cmp_nle_f32_e32 vcc, 1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e32 v0, 1.0, v0, vcc
; VI-SAFE-NEXT: v_xor_b32_e32 v0, 0x80000000, v0

; GCN-NONAN: v_max_f32_e64 v0, -v0, -1.0
define amdgpu_ps float @min_fneg_select_regression_0(float %a, float %b) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ult float %a, 1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float -1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}min_fneg_select_regression_posk_0:
; GCN-NOT: v_mul

; SI-SAFE: v_max_legacy_f32_e64 [[MIN:v[0-9]+]], 1.0, -v0

; VI-SAFE: v_cmp_nle_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e32 v0, -1.0, v0, vcc
; VI-SAFE-NEXT: v_xor_b32_e32 v0, 0x80000000, v0

; GCN-NONAN: v_max_f32_e64 v{{[0-9]+}}, -v0, 1.0
define amdgpu_ps float @min_fneg_select_regression_posk_0(float %a, float %b) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ult float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}max_fneg_select_regression_0:
; GCN-NOT: v_mul

; SI-SAFE: v_min_legacy_f32_e64 [[MIN:v[0-9]+]], -1.0, -v0

; VI-SAFE: v_cmp_nge_f32_e32 vcc, 1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e32 v0, 1.0, v0, vcc
; VI-SAFE-NEXT: v_xor_b32_e32 v0, 0x80000000, v0

; GCN-NONAN: v_min_f32_e64 [[MIN:v[0-9]+]], -v0, -1.0
define amdgpu_ps float @max_fneg_select_regression_0(float %a) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ugt float %a, 1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float -1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}max_fneg_select_regression_posk_0:
; GCN-NOT: v_mul

; SI-SAFE: v_min_legacy_f32_e64 [[MIN:v[0-9]+]], 1.0, -v0

; VI-SAFE: v_cmp_nge_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e32 v0, -1.0, v0, vcc
; VI-SAFE-NEXT: v_xor_b32_e32 v0, 0x80000000, v0

; GCN-NONAN: v_min_f32_e64 [[MIN:v[0-9]+]], -v0, 1.0
define amdgpu_ps float @max_fneg_select_regression_posk_0(float %a) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ugt float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
