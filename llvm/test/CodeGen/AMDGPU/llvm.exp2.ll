;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=EG --check-prefix=FUNC
;RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=CM --check-prefix=FUNC
;RUN: llc < %s -march=amdgcn | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=SI --check-prefix=FUNC
;RUN: llc < %s -march=amdgcn -mcpu=tonga | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=SI --check-prefix=FUNC

;FUNC-LABEL: {{^}}test:
;EG: EXP_IEEE
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;SI: v_exp_f32

define amdgpu_kernel void @test(ptr addrspace(1) %out, float %in) {
entry:
   %0 = call float @llvm.exp2.f32(float %in)
   store float %0, ptr addrspace(1) %out
   ret void
}

;FUNC-LABEL: {{^}}testv2:
;EG: EXP_IEEE
;EG: EXP_IEEE
; FIXME: We should be able to merge these packets together on Cayman so we
; have a maximum of 4 instructions.
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;SI: v_exp_f32
;SI: v_exp_f32

define amdgpu_kernel void @testv2(ptr addrspace(1) %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.exp2.v2f32(<2 x float> %in)
  store <2 x float> %0, ptr addrspace(1) %out
  ret void
}

;FUNC-LABEL: {{^}}testv4:
;EG: EXP_IEEE
;EG: EXP_IEEE
;EG: EXP_IEEE
;EG: EXP_IEEE
; FIXME: We should be able to merge these packets together on Cayman so we
; have a maximum of 4 instructions.
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}} (MASKED)
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;CM-DAG: EXP_IEEE T{{[0-9]+\.[XYZW]}}
;SI: v_exp_f32
;SI: v_exp_f32
;SI: v_exp_f32
;SI: v_exp_f32
define amdgpu_kernel void @testv4(ptr addrspace(1) %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.exp2.v4f32(<4 x float> %in)
  store <4 x float> %0, ptr addrspace(1) %out
  ret void
}

declare float @llvm.exp2.f32(float) readnone
declare <2 x float> @llvm.exp2.v2f32(<2 x float>) readnone
declare <4 x float> @llvm.exp2.v4f32(<4 x float>) readnone
