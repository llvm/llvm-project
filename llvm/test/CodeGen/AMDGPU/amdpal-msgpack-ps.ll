; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -enable-var-scope %s

; amdpal pixel shader: check for 0x2c0a (SPI_SHADER_PGM_RSRC1_PS) in pal
; metadata. Check for 0x2c0b (SPI_SHADER_PGM_RSRC2_PS) in pal metadata, and
; it has a value starting 0x42 as it is set to 0x42000000 in the metadata
; below. Also check that .internal_pipeline_hash is propagated.
; GCN-LABEL: {{^}}ps_amdpal:
; GCN: .amdgpu_pal_metadata
; GCN:         .internal_pipeline_hash:
; GCN-NEXT:      - 0x123456789abcdef0
; GCN-NEXT:      - 0xfedcba9876543210
; GCN:         .registers:
; GCN:           '0x2c0a (SPI_SHADER_PGM_RSRC1_PS)':
; GCN:           '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x42
define amdgpu_ps half @ps_amdpal(half %arg0) {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdgpu.pal.metadata.msgpack represents this:
;
; 	.amdgpu_pal_metadata
; ---
; amdpal.pipelines:
;   - .internal_pipeline_hash:
;       - 0x123456789abcdef0
;       - 0xfedcba9876543210
;     .registers:
;       '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x42000000
; ...
; 	.end_amdgpu_pal_metadata

!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!"\81\b0\61\6d\64\70\61\6c\2e\70\69\70\65\6c\69\6e\65\73\91\82\b7\2e\69\6e\74\65\72\6e\61\6c\5f\70\69\70\65\6c\69\6e\65\5f\68\61\73\68\92\cf\12\34\56\78\9a\bc\de\f0\cf\fe\dc\ba\98\76\54\32\10\aa\2e\72\65\67\69\73\74\65\72\73\81\cd\2c\0b\ce\42\00\00\00"};
