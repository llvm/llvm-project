; RUN: llc -mtriple=amdgcn < %s | FileCheck -enable-var-scope -check-prefixes=SI-SAFE,GCN %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -mtriple=amdgcn < %s | FileCheck -enable-var-scope --check-prefixes=GCN %s

; RUN: llc -mtriple=amdgcn -mcpu=fiji < %s | FileCheck -enable-var-scope -check-prefixes=VI-SAFE,GCN %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -mtriple=amdgcn -mcpu=fiji < %s | FileCheck -enable-var-scope --check-prefixes=GCN,VI-NNAN %s

; GCN-LABEL: {{^}}min_fneg_select_regression_0:
; GCN-NOT: v_mul

; SI: v_max_legacy_f32_e64 [[MIN:v[0-9]+]], -1.0, -v0

; VI-SAFE: v_cmp_nle_f32_e32 vcc, 1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, -1.0, -v0, vcc
define amdgpu_ps float @min_fneg_select_regression_0(float %a, float %b) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ult float %a, 1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float -1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}min_fneg_select_regression_posk_0:
; GCN-NOT: v_mul

; SI: v_max_legacy_f32_e64 [[MIN:v[0-9]+]], 1.0, -v0

; VI-SAFE: v_cmp_nle_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; VI-NNAN: v_max_f32_e64 v{{[0-9]+}}, -v0, 1.0
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
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, -1.0, -v0, vcc

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
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; GCN-NONAN: v_min_f32_e64 [[MIN:v[0-9]+]], -v0, 1.0
define amdgpu_ps float @max_fneg_select_regression_posk_0(float %a) #0 {
  %fneg.a = fsub float -0.0, %a
  %cmp.a = fcmp ugt float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ugt_a_neg1:
; SI: v_min_legacy_f32_e64 v0, 1.0, -v0

; VI-SAFE: v_cmp_nge_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; VI-NNAN: v_min_f32_e64 v0, -v0, 1.0
define amdgpu_ps float @select_fneg_a_or_q_cmp_ugt_a_neg1(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ugt float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ult_a_neg1:
; SI: v_max_legacy_f32_e64 v0, 1.0, -v0

; VI-SAFE: v_cmp_nle_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; VI-NNAN: v_max_f32_e64 v0, -v0, 1.0
define amdgpu_ps float @select_fneg_a_or_q_cmp_ult_a_neg1(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ult float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ogt_a_neg1:
; SI: v_min_legacy_f32_e64 v0, -v0, 1.0

; VI-SAFE: v_cmp_lt_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; VI-NNAN: v_min_f32_e64 v0, -v0, 1.0
define amdgpu_ps float @select_fneg_a_or_q_cmp_ogt_a_neg1(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ogt float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_olt_a_neg1:
; SI: v_max_legacy_f32_e64 v0, -v0, 1.0

; VI-SAFE: v_cmp_gt_f32_e32 vcc, -1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, 1.0, -v0, vcc

; VI-NANN: v_max_f32_e64 v0, -v0, 1.0
define amdgpu_ps float @select_fneg_a_or_q_cmp_olt_a_neg1(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp olt float %a, -1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ugt_a_neg8:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI-NEXT: v_min_legacy_f32_e64 v0, [[K]], -v0

; VI-SAFE-DAG: s_mov_b32 [[K0:s[0-9]+]], 0xc1000000
; VI-SAFE-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x41000000
; VI-SAFE: v_cmp_nge_f32_e32 vcc, [[K0]], v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, [[K1]], -v0, vcc

; VI-NNAN: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; VI-NNAN-NEXT: v_min_f32_e64 v0, -v0, [[K]]
define amdgpu_ps float @select_fneg_a_or_q_cmp_ugt_a_neg8(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ugt float %a, -8.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 8.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ult_a_neg8:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI-NEXT: v_max_legacy_f32_e64 v0, [[K]], -v0

; VI-SAFE-DAG: s_mov_b32 [[K0:s[0-9]+]], 0xc1000000
; VI-SAFE-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x41000000
; VI-SAFE: v_cmp_nle_f32_e32 vcc, [[K0]], v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, [[K1]], -v0, vcc

; VI-NNAN: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; VI-NNAN-NEXT: v_max_f32_e64 v0, -v0, [[K]]
define amdgpu_ps float @select_fneg_a_or_q_cmp_ult_a_neg8(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ult float %a, -8.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 8.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_ogt_a_neg8:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI-NEXT: v_min_legacy_f32_e64 v0, -v0, [[K]]

; VI-SAFE-DAG: s_mov_b32 [[K0:s[0-9]+]], 0xc1000000
; VI-SAFE-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x41000000
; VI-SAFE: v_cmp_lt_f32_e32 vcc, [[K0]], v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, [[K1]], -v0, vcc

; VI-NNAN: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; VI-NNAN-NEXT: v_min_f32_e64 v0, -v0, [[K]]
define amdgpu_ps float @select_fneg_a_or_q_cmp_ogt_a_neg8(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ogt float %a, -8.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 8.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_q_cmp_olt_a_neg8:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI-NEXT: v_max_legacy_f32_e64 v0, -v0, [[K]]


; VI-SAFE-DAG: s_mov_b32 [[K0:s[0-9]+]], 0xc1000000
; VI-SAFE-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x41000000
; VI-SAFE: v_cmp_gt_f32_e32 vcc, [[K0]], v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, [[K1]], -v0, vcc

; VI-NNAN: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; VI-NNAN-NEXT: v_max_f32_e64 v0, -v0, [[K]]
define amdgpu_ps float @select_fneg_a_or_q_cmp_olt_a_neg8(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp olt float %a, -8.0
  %min.a = select i1 %cmp.a, float %fneg.a, float 8.0
  ret float %min.a
}

; GCN-LABEL: {{^}}select_fneg_a_or_neg1_cmp_olt_a_1:
; SI: v_max_legacy_f32_e64 v0, -v0, -1.0

; VI-SAFE: v_cmp_gt_f32_e32 vcc, 1.0, v0
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, -1.0, -v0, vcc

; VI-NNAN: v_max_f32_e64 v0, -v0, -1.0
define amdgpu_ps float @select_fneg_a_or_neg1_cmp_olt_a_1(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp olt float %a, 1.0
  %min.a = select i1 %cmp.a, float %fneg.a, float -1.0
  ret float %min.a
}

; GCN-LABEL: {{^}}ult_a_select_fneg_a_b:
; SI: v_cmp_nge_f32_e32 vcc, v0, v1
; SI-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc

; VI-SAFE: v_cmp_nge_f32_e32 vcc, v0, v1
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc

; VI-NNAN: v_cmp_lt_f32_e32 vcc, v0, v1
; VI-NNAN-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc
define amdgpu_ps float @ult_a_select_fneg_a_b(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ult float %a, %b
  %min.a = select i1 %cmp.a, float %fneg.a, float %b
  ret float %min.a
}

; GCN-LABEL: {{^}}ugt_a_select_fneg_a_b:
; SI: v_cmp_nle_f32_e32 vcc, v0, v1
; SI-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc

; VI-SAFE: v_cmp_nle_f32_e32 vcc, v0, v1
; VI-SAFE-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc

; VI-NNAN: v_cmp_gt_f32_e32 vcc, v0, v1
; VI-NNAN-NEXT: v_cndmask_b32_e64 v0, v1, -v0, vcc
define amdgpu_ps float @ugt_a_select_fneg_a_b(float %a, float %b) #0 {
  %fneg.a = fneg float %a
  %cmp.a = fcmp ugt float %a, %b
  %min.a = select i1 %cmp.a, float %fneg.a, float %b
  ret float %min.a
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
