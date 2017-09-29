; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; amdpal pixel shader: check for 45864 (SPI_SHADER_PGM_RSRC1_ES) in .AMDGPU.config
; GCN-LABEL: .AMDGPU.config
; GCN: .long  45864
; GCN-LABEL: {{^}}es_amdpal:
define amdgpu_es half @es_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}


