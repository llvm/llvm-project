; RUN: llc -mtriple=amdgcn--amdpal -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -enable-var-scope %s

; amdpal compute shader: check for 0x2e12 (COMPUTE_PGM_RSRC1) in pal metadata
; SI-DAG: 0x2e12 (COMPUTE_PGM_RSRC1): 0x2c0000{{$}}
; VI-DAG: 0x2e12 (COMPUTE_PGM_RSRC1): 0x2c02c0{{$}}
; GFX9-DAG: 0x2e12 (COMPUTE_PGM_RSRC1): 0x2c0000{{$}}
define amdgpu_cs half @cs_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal evaluation shader: check for 0x2cca (SPI_SHADER_PGM_RSRC1_ES) in pal metadata
; SI-DAG: 0x2cca (SPI_SHADER_PGM_RSRC1_ES): 0x2c0000{{$}}
; VI-DAG: 0x2cca (SPI_SHADER_PGM_RSRC1_ES): 0x2c02c0{{$}}
; GFX9-DAG: 0x2cca (SPI_SHADER_PGM_RSRC1_ES): 0x2c0000{{$}}
define amdgpu_es half @es_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal geometry shader: check for 0x2c8a (SPI_SHADER_PGM_RSRC1_GS) in pal metadata
; SI-DAG: 0x2c8a (SPI_SHADER_PGM_RSRC1_GS): 0x2c0000{{$}}
; VI-DAG: 0x2c8a (SPI_SHADER_PGM_RSRC1_GS): 0x2c02c0{{$}}
; GFX9-DAG: 0x2c8a (SPI_SHADER_PGM_RSRC1_GS): 0x2c0000{{$}}
define amdgpu_gs half @gs_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal hull shader: check for 0x2d0a (SPI_SHADER_PGM_RSRC1_HS) in pal metadata
; SI-DAG: 0x2d0a (SPI_SHADER_PGM_RSRC1_HS): 0x2c0000{{$}}
; VI-DAG: 0x2d0a (SPI_SHADER_PGM_RSRC1_HS): 0x2c02c0{{$}}
; GFX9-DAG: 0x2d0a (SPI_SHADER_PGM_RSRC1_HS): 0x2c0000{{$}}
define amdgpu_hs half @hs_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal load shader: check for 0x2d4a (SPI_SHADER_PGM_RSRC1_LS) in pal metadata
; SI-DAG: 0x2d4a (SPI_SHADER_PGM_RSRC1_LS): 0x2c0000{{$}}
; VI-DAG: 0x2d4a (SPI_SHADER_PGM_RSRC1_LS): 0x2c02c0{{$}}
; GFX9-DAG: 0x2d4a (SPI_SHADER_PGM_RSRC1_LS): 0x2c0000{{$}}
define amdgpu_ls half @ls_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal pixel shader: check for 0x2c0a (SPI_SHADER_PGM_RSRC1_PS) in pal metadata
; below.
; SI-DAG:           0x2c0a (SPI_SHADER_PGM_RSRC1_PS): 0x2c0000{{$}}
; VI-DAG:           0x2c0a (SPI_SHADER_PGM_RSRC1_PS): 0x2c02c0{{$}}
; GFX9-DAG:         0x2c0a (SPI_SHADER_PGM_RSRC1_PS): 0x2c0000{{$}}
define amdgpu_ps half @ps_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

; amdpal vertex shader: check for 45352 (SPI_SHADER_PGM_RSRC1_VS) in pal metadata
; SI-DAG: 0x2c4a (SPI_SHADER_PGM_RSRC1_VS): 0x2c0000{{$}}
; VI-DAG: 0x2c4a (SPI_SHADER_PGM_RSRC1_VS): 0x2c02c0{{$}}
; GFX9-DAG: 0x2c4a (SPI_SHADER_PGM_RSRC1_VS): 0x2c0000{{$}}
define amdgpu_vs half @vs_amdpal(half %arg0) #0 {
  %add = fadd half %arg0, 1.0
  ret half %add
}

attributes #0 = { "denormal-fp-math-f32"="preserve-sign,preserve-sign" }

; amdgpu.pal.metadata.msgpack represents this:
;
; 	.amdgpu_pal_metadata
; ---
; amdpal.pipelines:
;   - .internal_pipeline_hash:
;       - 0x123456789abcdef0
;       - 0xfedcba9876543210
;     .registers:
;       0x2c0b (SPI_SHADER_PGM_RSRC2_PS): 0x42000000
; ...
; 	.end_amdgpu_pal_metadata

!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!"\81\b0\61\6d\64\70\61\6c\2e\70\69\70\65\6c\69\6e\65\73\91\82\b7\2e\69\6e\74\65\72\6e\61\6c\5f\70\69\70\65\6c\69\6e\65\5f\68\61\73\68\92\cf\12\34\56\78\9a\bc\de\f0\cf\fe\dc\ba\98\76\54\32\10\aa\2e\72\65\67\69\73\74\65\72\73\81\cd\2c\0b\ce\42\00\00\00"};
