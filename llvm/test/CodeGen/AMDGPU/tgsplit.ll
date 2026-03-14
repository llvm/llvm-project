; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a < %s | FileCheck -check-prefixes=GCN,NOTGSPLIT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -mattr=+tgsplit < %s | FileCheck -check-prefixes=GCN,TGSPLIT %s

; GCN-LABEL: .amdhsa_kernel test
; NOTGSPLIT: .amdhsa_tg_split 0
; NOTGSPLIT: COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
; TGSPLIT:   .amdhsa_tg_split 1
; TGSPLIT:   COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 1
define amdgpu_kernel void @test() {
  ret void
}
