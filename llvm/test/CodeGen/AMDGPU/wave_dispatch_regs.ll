; RUN: llc -global-isel=1 -mtriple=amdgcn--amdpal < %s | FileCheck -check-prefix=GCN -check-prefix=SI -enable-var-scope %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdpal -mcpu=tonga < %s | FileCheck -check-prefix=GCN -check-prefix=VI -enable-var-scope %s
; RUN: llc -global-isel=1 -mtriple=amdgcn--amdpal -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s
; RUN: llc -global-isel=0 -mtriple=amdgcn--amdpal < %s | FileCheck -check-prefix=GCN -check-prefix=SI -enable-var-scope %s
; RUN: llc -global-isel=0 -mtriple=amdgcn--amdpal -mcpu=tonga < %s | FileCheck -check-prefix=GCN -check-prefix=VI -enable-var-scope %s
; RUN: llc -global-isel=0 -mtriple=amdgcn--amdpal -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; This compute shader has input args that claim that it has 17 sgprs and 5 vgprs
; in wave dispatch. Ensure that the sgpr and vgpr counts in COMPUTE_PGM_RSRC1
; are set to reflect that, even though the registers are not used in the shader.

; GCN-LABEL: {{^}}_amdgpu_cs_main:
; GCN:         .amdgpu_pal_metadata
; GCN-NEXT: ---
; GCN-NEXT: amdpal.pipelines:
; GCN-NEXT:   - .hardware_stages:
; GCN-NEXT:       .cs:
; GCN-NEXT:         .entry_point:    _amdgpu_cs_main
; GCN-NEXT:         .entry_point_symbol:    _amdgpu_cs_main
; GCN-NEXT:         .scratch_memory_size: 0
; SI-NEXT:          .sgpr_count:     0x11
; VI-NEXT:          .sgpr_count:     0x60
; GFX9-NEXT:        .sgpr_count:     0x15
; SI-NEXT:          .vgpr_count:     0x5
; VI-NEXT:          .vgpr_count:     0x5
; GFX9-NEXT:        .vgpr_count:     0x5
; GCN-NEXT:     .registers:
; SI-NEXT:        '0x2e12 (COMPUTE_PGM_RSRC1)': 0x{{[0-9a-f]*}}81
; VI-NEXT:        '0x2e12 (COMPUTE_PGM_RSRC1)': 0x{{[0-9a-f]*}}c1
; GFX9-NEXT:      '0x2e12 (COMPUTE_PGM_RSRC1)': 0x{{[0-9a-f]*}}81
; GCN-NEXT:       '0x2e13 (COMPUTE_PGM_RSRC2)': 0
; GCN-NEXT: ...
; GCN-NEXT:         .end_amdgpu_pal_metadata

define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg, i32 inreg, <2 x i32> inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, i32 inreg, <3 x i32> inreg, i32 inreg, <5 x i32>) {
.entry:
  ret void
}
