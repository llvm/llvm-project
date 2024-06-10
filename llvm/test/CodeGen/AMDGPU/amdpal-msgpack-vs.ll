; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal vertex shader: check for 45352 (SPI_SHADER_PGM_RSRC1_VS) in pal metadata
; GCN-LABEL: {{^}}vs_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2c4a (SPI_SHADER_PGM_RSRC1_VS)'
define amdgpu_vs half @vs_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
