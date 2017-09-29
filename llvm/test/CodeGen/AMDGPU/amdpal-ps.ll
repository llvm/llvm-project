; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; amdpal pixel shader: check for 45096 (SPI_SHADER_PGM_RSRC1_PS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45096
; GCN-LABEL: {{^}}ps_amdpal:
define amdgpu_ps half @ps_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}


