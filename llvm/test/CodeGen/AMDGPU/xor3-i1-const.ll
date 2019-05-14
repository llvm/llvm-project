; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; This test used to crash
; GCN-LABEL: {{^}}xor3_i1_const:
; GCN: s_xor_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], -1
; GCN: s_xor_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], -1
define amdgpu_ps float @xor3_i1_const(float inreg %arg1, i32 inreg %arg2) {
main_body:
  %tmp26 = fcmp nsz olt float %arg1, 0.000000e+00
  %tmp28 = call nsz float @llvm.amdgcn.interp.p2(float undef, float undef, i32 0, i32 0, i32 %arg2)
  %tmp29 = fcmp nsz olt float %arg1, 5.700000e+01
  %tmp31 = fcmp nsz olt float %tmp28, 0.000000e+00
  %.demorgan = and i1 %tmp26, %tmp29
  %tmp34 = xor i1 %.demorgan, true
  %tmp35 = and i1 %tmp31, %tmp34
  %tmp36 = xor i1 %tmp35, true
  %tmp37 = xor i1 %.demorgan, %tmp36
  %tmp42 = or i1 %tmp37, %tmp35
  %tmp43 = select i1 %tmp42, float 1.000000e+00, float 0.000000e+00
  ret float %tmp43
}

declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32)
