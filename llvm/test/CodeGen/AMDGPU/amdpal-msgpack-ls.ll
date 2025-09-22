; RUN: llc -mtriple=amdgcn--amdpal < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga < %s | FileCheck -check-prefix=GCN %s

; amdpal load shader: check for 0x2d4a (SPI_SHADER_PGM_RSRC1_LS) in pal metadata
; GCN-LABEL: {{^}}ls_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2d4a (SPI_SHADER_PGM_RSRC1_LS)'
define amdgpu_ls half @ls_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
