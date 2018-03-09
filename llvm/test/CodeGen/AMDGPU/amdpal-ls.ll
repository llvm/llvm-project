; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; amdpal load shader: check for 0x2d4a (SPI_SHADER_PGM_RSRC1_LS) in pal metadata
; GCN-LABEL: {{^}}ls_amdpal:
; GCN: .amd_amdgpu_pal_metadata{{.*}}0x2d4a,
define amdgpu_ls half @ls_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}


