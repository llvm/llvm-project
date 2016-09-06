; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; These tests check that fdiv is expanded correctly and also test that the
; scheduler is scheduling the RECIP_IEEE and MUL_IEEE instructions in separate
; instruction groups.

; These test check that fdiv using unsafe_fp_math, coarse fp div, and IEEE754 fp div.

; FUNC-LABEL: {{^}}fdiv_f32:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[2].W
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, PS

; SI: v_div_scale_f32
; SI-DAG: v_div_scale_f32

; SI-DAG: v_rcp_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_mul_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_fma_f32
; SI: v_div_fmas_f32
; SI: v_div_fixup_f32
define void @fdiv_f32(float addrspace(1)* %out, float %a, float %b) #0 {
entry:
  %fdiv = fdiv float %a, %b
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_25ulp_f32:
; SI: v_cndmask_b32
; SI: v_mul_f32
; SI: v_rcp_f32
; SI: v_mul_f32
; SI: v_mul_f32
define void @fdiv_25ulp_f32(float addrspace(1)* %out, float %a, float %b) #0 {
entry:
  %fdiv = fdiv float %a, %b, !fpmath !0
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; Use correct fdiv
; FUNC-LABEL: {{^}}fdiv_25ulp_denormals_f32:
; SI: v_fma_f32
; SI: v_div_fmas_f32
; SI: v_div_fixup_f32
define void @fdiv_25ulp_denormals_f32(float addrspace(1)* %out, float %a, float %b) #2 {
entry:
  %fdiv = fdiv float %a, %b, !fpmath !0
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_fast_denormals_f32:
; SI: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; SI: v_mul_f32_e32 [[RESULT:v[0-9]+]], s{{[0-9]+}}, [[RCP]]
; SI-NOT: [[RESULT]]
; SI: buffer_store_dword [[RESULT]]
define void @fdiv_fast_denormals_f32(float addrspace(1)* %out, float %a, float %b) #2 {
entry:
  %fdiv = fdiv fast float %a, %b
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_f32_fast_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[2].W
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, PS

; SI: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; SI: v_mul_f32_e32 [[RESULT:v[0-9]+]], s{{[0-9]+}}, [[RCP]]
; SI-NOT: [[RESULT]]
; SI: buffer_store_dword [[RESULT]]
define void @fdiv_f32_fast_math(float addrspace(1)* %out, float %a, float %b) #0 {
entry:
  %fdiv = fdiv fast float %a, %b
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_f32_arcp_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[2].W
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, PS

; SI: v_rcp_f32_e32 [[RCP:v[0-9]+]], s{{[0-9]+}}
; SI: v_mul_f32_e32 [[RESULT:v[0-9]+]], s{{[0-9]+}}, [[RCP]]
; SI-NOT: [[RESULT]]
; SI: buffer_store_dword [[RESULT]]
define void @fdiv_f32_arcp_math(float addrspace(1)* %out, float %a, float %b) #0 {
entry:
  %fdiv = fdiv arcp float %a, %b
  store float %fdiv, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; SI: v_div_scale_f32
; SI: v_div_scale_f32
; SI: v_div_scale_f32
; SI: v_div_scale_f32
define void @fdiv_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #0 {
entry:
  %fdiv = fdiv <2 x float> %a, %b
  store <2 x float> %fdiv, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_ulp25_v2f32:
; SI: v_cmp_gt_f32
; SI: v_cmp_gt_f32
define void @fdiv_ulp25_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #0 {
entry:
  %fdiv = fdiv arcp <2 x float> %a, %b, !fpmath !0
  store <2 x float> %fdiv, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32_fast_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; SI: v_rcp_f32
; SI: v_rcp_f32
define void @fdiv_v2f32_fast_math(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #0 {
entry:
  %fdiv = fdiv fast <2 x float> %a, %b
  store <2 x float> %fdiv, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32_arcp_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; SI: v_rcp_f32
; SI: v_rcp_f32
define void @fdiv_v2f32_arcp_math(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) #0 {
entry:
  %fdiv = fdiv arcp <2 x float> %a, %b
  store <2 x float> %fdiv, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v4f32:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS

; SI: v_div_fixup_f32
; SI: v_div_fixup_f32
; SI: v_div_fixup_f32
; SI: v_div_fixup_f32
define void @fdiv_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) #0 {
  %b_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float>, <4 x float> addrspace(1) * %in
  %b = load <4 x float>, <4 x float> addrspace(1) * %b_ptr
  %result = fdiv <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v4f32_fast_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS

; SI: v_rcp_f32
; SI: v_rcp_f32
; SI: v_rcp_f32
; SI: v_rcp_f32
define void @fdiv_v4f32_fast_math(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) #0 {
  %b_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float>, <4 x float> addrspace(1) * %in
  %b = load <4 x float>, <4 x float> addrspace(1) * %b_ptr
  %result = fdiv fast <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v4f32_arcp_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}, PS

; SI: v_rcp_f32
; SI: v_rcp_f32
; SI: v_rcp_f32
; SI: v_rcp_f32
define void @fdiv_v4f32_arcp_math(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) #0 {
  %b_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float>, <4 x float> addrspace(1) * %in
  %b = load <4 x float>, <4 x float> addrspace(1) * %b_ptr
  %result = fdiv arcp <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind "enable-unsafe-fp-math"="false" "target-features"="-fp32-denormals" }
attributes #1 = { nounwind "enable-unsafe-fp-math"="true" "target-features"="-fp32-denormals" }
attributes #2 = { nounwind "enable-unsafe-fp-math"="false" "target-features"="+fp32-denormals" }

!0 = !{float 2.500000e+00}
