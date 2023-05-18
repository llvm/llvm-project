; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal pixel shader: check for 0x2c0a (SPI_SHADER_PGM_RSRC1_PS) in pal
; metadata. Check for 0x2c0b (SPI_SHADER_PGM_RSRC2_PS) in pal metadata, and
; it has a value starting 0x42 as it is set to 0x42000000 in the metadata
; below. Also check that key 0x10000000 value 0x12345678 is propagated.
; GCN-LABEL: {{^}}ps_amdpal:
; GCN: .amd_amdgpu_pal_metadata{{.*0x2c0a,[^,]*,0x2c0b,0x42.*,0x10000000,0x12345678}}
define amdgpu_ps half @ps_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

!amdgpu.pal.metadata = !{!0}
!0 = !{i32 11275, i32 1107296256, i32 268435456, i32 305419896}
