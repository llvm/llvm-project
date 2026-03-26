; Test failure to generate the _dvgpr$ symbol for an anonymous amdgpu_cs_chain function with dynamic vgprs.

; RUN: not llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s 2>&1 | FileCheck -check-prefixes=ERR %s

; Anonymous function with 129 VGPRs, which is too many with a block size of 16.
;
; ERR-DAG: .set __unnamed_1.num_vgpr, 129
; ERR-DAG: too many DVGPR blocks for _dvgpr$ symbol for '__unnamed_1'
;
define amdgpu_cs_chain void @0(<121 x float> %arg) #0 {
  tail call void @0(<121 x float> %arg)
  ret void
}

; Function that is OK, that chains to @1.
;
define amdgpu_cs_chain void @funcOk(<16 x float> %arg) {
  %vec87 = shufflevector <16 x float> %arg, <16 x float> %arg, <121 x i32> splat(i32 0)
  tail call void @0(<121 x float> %vec87)
  ret void
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }

