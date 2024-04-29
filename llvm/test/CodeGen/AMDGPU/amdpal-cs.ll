; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; GCN-LABEL: {{^}}cs_amdpal:
; GCN:           .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .hardware_stages:
; GCN-NEXT:       .cs:
; GCN-NEXT:         .entry_point:    cs_amdpal
; GCN-NEXT:         .scratch_memory_size: 0
; GCN:     .registers:
; GCN-NEXT:       '0x2e12 (COMPUTE_PGM_RSRC1)':
; GCN-NEXT:       '0x2e13 (COMPUTE_PGM_RSRC2)':
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata
define amdgpu_cs half @cs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}
