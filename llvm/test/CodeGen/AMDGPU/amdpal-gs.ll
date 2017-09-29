; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; amdpal pixel shader: check for 45608 (SPI_SHADER_PGM_RSRC1_GS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45608
; GCN-LABEL: {{^}}gs_amdpal:
define amdgpu_gs half @gs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}


