; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -enable-var-scope %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s
; RUN: llc -global-isel -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; GCN-LABEL: {{^}}gfx_callable_amdpal:
; GCN:         .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .registers:      {}
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata
define amdgpu_gfx half @gfx_callable_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}
