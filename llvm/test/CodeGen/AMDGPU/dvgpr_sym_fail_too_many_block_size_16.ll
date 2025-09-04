; Test failure to generate the _dvgpr$ symbol for an amdgpu_cs_chain function with dynamic vgprs.

; RUN: not llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s 2>&1 | FileCheck -check-prefixes=ERR %s

; Function with 129 VGPRs, which is too many with a block size of 16.
;
; ERR-DAG: .set func129.num_vgpr, 129
; ERR-DAG: too many DVGPR blocks for _dvgpr$ symbol for 'func129'
;
define amdgpu_cs_chain void @func129(<121 x float> %arg) #0 {
  tail call void @func129(<121 x float> %arg)
  ret void
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }
