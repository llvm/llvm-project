; RUN: llc -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s


declare <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32)

; GCN-LABEL: {{^}}water_loop_rsrc:

; GCN: [[RSRC_LOOP:[a-zA-Z0-9_]+]]:                                  ; =>This Inner Loop Header: Depth=1
; GCN-NEXT: v_readfirstlane_b32 s[[SREG0:[0-9]+]], v[[VREG0:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG1:[0-9]+]], v[[VREG1:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG2:[0-9]+]], v[[VREG2:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG3:[0-9]+]], v[[VREG3:[0-9]+]]
; GCN-NEXT: v_cmp_eq_u64_e32 [[CMP0:vcc]], s{{\[}}[[SREG0]]:[[SREG1]]{{\]}}, v{{\[}}[[VREG0]]:[[VREG1]]{{\]}}
; GCN-NEXT: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SREG2]]:[[SREG3]]{{\]}}, v{{\[}}[[VREG2]]:[[VREG3]]{{\]}}
; GCN-NEXT: v_readfirstlane_b32 s[[SREG4:[0-9]+]], v[[VREG4:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG5:[0-9]+]], v[[VREG5:[0-9]+]]
; GCN-NEXT: s_and_b64 [[AND0:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; GCN-NEXT: v_cmp_eq_u64_e32 [[CMP2:vcc]], s{{\[}}[[SREG4]]:[[SREG5]]{{\]}}, v{{\[}}[[VREG4]]:[[VREG5]]{{\]}}
; GCN-NEXT: v_readfirstlane_b32 s[[SREG6:[0-9]+]], v[[VREG6:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG7:[0-9]+]], v[[VREG7:[0-9]+]]
; GCN-NEXT: v_cmp_eq_u64_e64 [[CMP3:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SREG6]]:[[SREG7]]{{\]}}, v{{\[}}[[VREG6]]:[[VREG7]]{{\]}}
; GCN-NEXT: s_and_b64 [[AND1:s\[[0-9]+:[0-9]+\]]], [[AND0]], [[CMP2]]
; GCN-NEXT: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[AND1]], [[CMP3]]
; GCN-NEXT: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; GCN-NEXT: s_nop 0
; GCN-NEXT: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, s{{\[}}[[SREG0]]:[[SREG7]]{{\]}}, {{s\[[0-9]+:[0-9]+\]}} dmask:0x1
; GCN-NEXT: ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3_vgpr4_vgpr5_vgpr6_vgpr7
; GCN-NEXT: ; implicit-def: $vgpr8_vgpr9
; GCN-NEXT: s_xor_b64 exec, exec, [[SAVE]]
; GCN-NEXT: s_cbranch_execnz [[RSRC_LOOP]]
define amdgpu_ps <4 x float> @water_loop_rsrc(<8 x i32> %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32 1, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}


; GCN-LABEL: {{^}}water_loop_samp:

; GCN: [[SAMP_LOOP:[a-zA-Z0-9_]+]]:                                  ; =>This Inner Loop Header: Depth=1
; GCN-NEXT: v_readfirstlane_b32 s[[SREG0:[0-9]+]], v[[VREG0:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG1:[0-9]+]], v[[VREG1:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG2:[0-9]+]], v[[VREG2:[0-9]+]]
; GCN-NEXT: v_readfirstlane_b32 s[[SREG3:[0-9]+]], v[[VREG3:[0-9]+]]

; GCN-NEXT: v_cmp_eq_u64_e32 [[CMP0:vcc]], s{{\[}}[[SREG0]]:[[SREG1]]{{\]}}, v{{\[}}[[VREG0]]:[[VREG1]]{{\]}}
; GCN-NEXT: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SREG2]]:[[SREG3]]{{\]}}, v{{\[}}[[VREG2]]:[[VREG3]]{{\]}}
; GCN-NEXT: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; GCN-NEXT: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; GCN-NEXT: s_nop 0

; GCN-NEXT: image_gather4 {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, s{{\[}}[[SREG0]]:[[SREG3]]{{\]}} dmask:0x1
; GCN-NEXT: ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
; GCN-NEXT: ; implicit-def: $vgpr4_vgpr5
; GCN-NEXT: s_xor_b64 exec, exec, [[SAVE]]
; GCN-NEXT: s_cbranch_execnz [[SAMP_LOOP]]
define amdgpu_ps <4 x float> @water_loop_samp(<8 x i32> inreg %rsrc, <4 x i32> %samp, float %s, float %t) {
main_body:
  %v = call <4 x float> @llvm.amdgcn.image.gather4.2d.v4f32.f32(i32 1, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 0, i32 0)
  ret <4 x float> %v
}
