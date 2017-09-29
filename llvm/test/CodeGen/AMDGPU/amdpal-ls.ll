; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; amdpal pixel shader: check for 46376 (SPI_SHADER_PGM_RSRC1_LS) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  46376
; GCN-LABEL: {{^}}ls_amdpal:
define amdgpu_ls half @ls_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}


