; Test generation of _dvgpr$ symbol for an amdgpu_cs_chain function with +dynamic-vgpr.

; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1200 < %s | FileCheck -check-prefixes=DVGPR %s

; Function with 0 VGPRs, which counts as 1 block.
;
; DVGPR-LABEL: func0:
; DVGPR: .set _dvgpr$func0, func0+0
;
define amdgpu_cs_chain void @func0() #0 {
  ret void
}

; Function with 21 VGPRs, which is 2 blocks.
;
; DVGPR-LABEL: func21:
; DVGPR: .set _dvgpr$func21, func21+8
;
define amdgpu_cs_chain void @func21(<21 x float> %arg) #0 {
  ret void
}

; Anonymous function with 87 VGPRs, which is 6 blocks.
;
; DVGPR: [[FUNC87:__unnamed[^:]*]]:
; DVGPR: .set _dvgpr$[[FUNC87]], [[FUNC87]]+40
;
define amdgpu_cs_chain void @0(<87 x float> %arg) #0 {
  ret void
}

; Function with 128 VGPRs, which is 8 blocks.
;
; DVGPR-LABEL: func128:
; DVGPR: .set _dvgpr$func128, func128+56
;
define amdgpu_cs_chain void @func128(<128 x float> %arg) #0 {
  %vec87 = shufflevector <128 x float> %arg, <128 x float> %arg, <87 x i32> splat(i32 0)
  tail call void @0(<87 x float> %vec87)
  ret void
}

; Function with 79 VGPRs, which is 3 blocks with a block size of 32.
;
; DVGPR-LABEL: func79:
; DVGPR: .set _dvgpr$func79, func79+16
;
define amdgpu_cs_chain void @func79(<79 x float> %arg) #1 {
  ret void
}

; Function with 225 VGPRs, which is 8 blocks with a block size of 32.
;
; DVGPR-LABEL: func225:
; DVGPR: .set _dvgpr$func225, func225+56
;
define amdgpu_cs_chain void @func225(<225 x float> %arg) #1 {
  ret void
}

attributes #0 = { "amdgpu-dynamic-vgpr-block-size"="16" }
attributes #1 = { "amdgpu-dynamic-vgpr-block-size"="32" }
