; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck -check-prefixes=GCN %s

; GCN-LABEL: .amdhsa_kernel test_notgsplit
; GCN: .amdhsa_tg_split 0
; GCN: COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
define amdgpu_kernel void @test_notgsplit() #0 {
  ret void
}

; GCN-LABEL: .amdhsa_kernel test_tgsplit
; GCN: .amdhsa_tg_split 1
; GCN: COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 1
define amdgpu_kernel void @test_tgsplit() #1 {
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "amdgpu-tg-split" }
