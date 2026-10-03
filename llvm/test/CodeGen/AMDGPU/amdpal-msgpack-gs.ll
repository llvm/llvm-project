; RUN: llc -mtriple=amdgpu6.00--amdpal < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.02--amdpal < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu9.00--amdpal < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal geometry shader: check for 0x2c8a (SPI_SHADER_PGM_RSRC1_GS) in pal metadata
; GCN-LABEL: {{^}}gs_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2c8a (SPI_SHADER_PGM_RSRC1_GS)'
define amdgpu_gs half @gs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
