; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; GCN-LABEL: {{^}}gs_amdpal:
; GCN:         .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .hardware_stages:
; GCN-NEXT:       .gs:
; GCN-NEXT:         .entry_point:    _amdgpu_gs
; GCN-NEXT:         .entry_point_symbol:    gs_amdpal
; GCN-NEXT:         .scratch_memory_size: 0
; GCN:     .registers:
; GCN-NEXT:       '0x2c8a (SPI_SHADER_PGM_RSRC1_GS)': 0
; GCN-NEXT:       '0x2c8b (SPI_SHADER_PGM_RSRC2_GS)': 0
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata
define amdgpu_gs half @gs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}
