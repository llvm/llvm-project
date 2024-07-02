; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; amdpal evaluation shader: check for 0x2cca (SPI_SHADER_PGM_RSRC1_ES) in pal metadata
; GCN-LABEL: {{^}}es_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN: '0x2cca (SPI_SHADER_PGM_RSRC1_ES)'
define amdgpu_es half @es_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; Force MsgPack format metadata
!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!""}
