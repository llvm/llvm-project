; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}ls_amdpal:
; GCN:         .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .hardware_stages:
; GCN-NEXT:       .ls:
; GCN-NEXT:         .entry_point:    _amdgpu_ls
; GCN-NEXT:         .entry_point_symbol:    ls_amdpal
; GCN-NEXT:         .scratch_memory_size: 0
; GCN:     .registers:
; GCN-NEXT:       '0x2d4a (SPI_SHADER_PGM_RSRC1_LS)': 0
; GCN-NEXT:       '0x2d4b (SPI_SHADER_PGM_RSRC2_LS)': 0
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata
define amdgpu_ls half @ls_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}
