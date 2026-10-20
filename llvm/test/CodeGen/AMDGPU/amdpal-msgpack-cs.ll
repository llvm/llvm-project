; RUN: llc -mtriple=amdgpu6.00--amdpal < %s | FileCheck -check-prefix=GCN -enable-var-scope %s
; RUN: llc -mtriple=amdgpu8.02--amdpal < %s | FileCheck -check-prefix=GCN -enable-var-scope %s
; RUN: llc -mtriple=amdgpu9.00--amdpal < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal compute shader: check for 0x2e12 (COMPUTE_PGM_RSRC1) in pal metadata
; GCN-LABEL: {{^}}cs_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2e12 (COMPUTE_PGM_RSRC1)'
define amdgpu_cs half @cs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
