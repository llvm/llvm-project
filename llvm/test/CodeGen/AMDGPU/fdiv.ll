; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -amdgpu-fast-fdiv < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=I754 %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs -enable-unsafe-fp-math < %s | FileCheck -check-prefix=UNSAFE-FP %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 %s

; These tests check that fdiv is expanded correctly and also test that the
; scheduler is scheduling the RECIP_IEEE and MUL_IEEE instructions in separate
; instruction groups.

; These test check that fdiv using unsafe_fp_math, coarse fp div, and IEEE754 fp div.

; FUNC-LABEL: {{^}}fdiv_f32:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32

; I754-DAG: v_div_scale_f32
; I754-DAG: v_rcp_f32
; I754-DAG: v_fma_f32
; I754-DAG: v_mul_f32
; I754-DAG: v_fma_f32
; I754-DAG: v_div_fixup_f32
define void @fdiv_f32(float addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fdiv float %a, %b
  store float %0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_f32_fast_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_f32_fast_math(float addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fdiv fast float %a, %b
  store float %0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_f32_arcp_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_f32_arcp_math(float addrspace(1)* %out, float %a, float %b) {
entry:
  %0 = fdiv arcp float %a, %b
  store float %0, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32

; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_fixup_f32
; I754: v_div_fixup_f32
define void @fdiv_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fdiv <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32_fast_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_v2f32_fast_math(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fdiv fast <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fdiv_v2f32_arcp_math:
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]+\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[3].X, PS
; R600-DAG: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].W, PS

; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_rcp_f32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_v2f32_arcp_math(<2 x float> addrspace(1)* %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fdiv arcp <2 x float> %a, %b
  store <2 x float> %0, <2 x float> addrspace(1)* %out
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

; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32

; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_scale_f32
; I754: v_div_fixup_f32
; I754: v_div_fixup_f32
; I754: v_div_fixup_f32
; I754: v_div_fixup_f32
define void @fdiv_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
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

; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_v4f32_fast_math(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
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

; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_rcp_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32
; UNSAFE-FP: v_mul_f32_e32

; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
; SI-DAG: v_rcp_f32
; SI-DAG: v_mul_f32
define void @fdiv_v4f32_arcp_math(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x float>, <4 x float> addrspace(1)* %in, i32 1
  %a = load <4 x float>, <4 x float> addrspace(1) * %in
  %b = load <4 x float>, <4 x float> addrspace(1) * %b_ptr
  %result = fdiv arcp <4 x float> %a, %b
  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}
