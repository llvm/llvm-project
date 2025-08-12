; Test failure to generate the _dvgpr$ symbol for an anonymous amdgpu_cs_chain function with dynamic vgprs.

; RUN: not llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s 2>&1 | FileCheck -check-prefixes=ERR %s

define amdgpu_cs_chain void @0(<87 x float> %arg) #0 {
  ret void
}

; Anonymous function with 129 VGPRs, which is too many with a block size of 16.
;
; ERR: too many DVGPR blocks for _dvgpr$ symbol for '__unnamed_2'
;
define amdgpu_cs_chain void @1(<129 x float> %arg) #0 {
  %vec87 = shufflevector <129 x float> %arg, <129 x float> %arg, <87 x i32> splat(i32 0)
  tail call void @0(<87 x float> %vec87)
  ret void
}

; Function that is OK, that chains to @1.
;
define amdgpu_cs_chain void @funcOk(<16 x float> %arg) #0 {
  %vec87 = shufflevector <16 x float> %arg, <16 x float> %arg, <129 x i32> splat(i32 0)
  tail call void @1(<129 x float> %vec87)
  ret void
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }

