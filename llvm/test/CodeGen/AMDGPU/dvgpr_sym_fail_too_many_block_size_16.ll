; Test failure to generate the _dvgpr$ symbol for an amdgpu_cs_chain function with dynamic vgprs.

; RUN: not llc -mtriple=amdgpu12.00-amd-amdpal < %s 2>&1 | FileCheck -check-prefixes=ERR %s

; Function with 129 VGPRs, which is too many with a block size of 16.
;
; ERR-DAG: .set .Lfunc129.num_vgpr, 129
; ERR-DAG: DVGPR block count 9 exceeds maximum of 8 for __dvgpr$ symbol for 'func129'
;
define amdgpu_cs_chain void @func129(<121 x float> %arg) #0 {
  tail call void @func129(<121 x float> %arg)
  ret void
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }
