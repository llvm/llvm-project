; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}main:
; GCN: v_movreld_b32_e32 v0,
; GCN: v_mov_b32_e32 v0, v1
; GCN: ; return
define amdgpu_ps float @main(i32 inreg %arg) #0 {
main_body:
  %tmp24 = insertelement <2 x float> undef, float 0.000000e+00, i32 %arg
  %tmp25 = extractelement <2 x float> %tmp24, i32 1
  ret float %tmp25
}

attributes #0 = { "InitialPSInputAddr"="36983" }
