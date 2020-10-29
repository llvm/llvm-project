; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; This pixel shader does not use the result of its interpolation, so it would
; end up with an interpolation mode set in PSAddr but not PSEnable. This test tests
; the workaround that ensures that an interpolation mode is also set in PSEnable.
; GCN-LABEL: {{^}}amdpal_psenable:
; GCN:         .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .hardware_stages:
; GCN-NEXT:       .ps:
; GCN-NEXT:         .entry_point:    amdpal_psenable
; GCN-NEXT:         .scratch_memory_size: 0
; GCN:     .registers:
; GCN-NEXT:       0x2c0a (SPI_SHADER_PGM_RSRC1_PS):
; GCN-NEXT:       0x2c0b (SPI_SHADER_PGM_RSRC2_PS):
; GCN-NEXT:       0xa1b3 (SPI_PS_INPUT_ENA): 0x2
; GCN-NEXT:       0xa1b4 (SPI_PS_INPUT_ADDR): 0x2
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata
define amdgpu_ps void @amdpal_psenable(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <2 x float> %pos) #6 {
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  ret void
}

declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #2
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #2

attributes #6 = { nounwind "InitialPSInputAddr"="2" }
