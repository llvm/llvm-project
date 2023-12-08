; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GPRIDX %s

; GCN-LABEL: {{^}}main:

; MOVREL: s_mov_b32 m0, s0
; MOVREL-NEXT: v_movreld_b32_e32 v0,

; GPRIDX: s_set_gpr_idx_on s0, gpr_idx(DST)
; GPRIDX-NEXT: v_mov_b32_e32 v0, 1.0
; GPRIDX-NEXT: s_set_gpr_idx_off

; GCN-NEXT: v_mov_b32_e32 v0, v1
; GCN-NEXT: ; return
define amdgpu_ps float @main(i32 inreg %arg) #0 {
main_body:
  %tmp24 = insertelement <16 x float> zeroinitializer, float 1.000000e+00, i32 %arg
  %tmp25 = extractelement <16 x float> %tmp24, i32 1
  ret float %tmp25
}

attributes #0 = { "InitialPSInputAddr"="36983" }
