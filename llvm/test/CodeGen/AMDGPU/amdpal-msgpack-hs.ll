; RUN: llc -mtriple=amdgpu6.00--amdpal < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.02--amdpal < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu9.00--amdpal < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal hull shader: check for 0x2d0a (SPI_SHADER_PGM_RSRC1_HS) in pal metadata
; GCN-LABEL: {{^}}hs_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2d0a (SPI_SHADER_PGM_RSRC1_HS)'
define amdgpu_hs half @hs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
