; RUN: llc -mtriple=amdgpu11.00--amdpal -verify-misched -amdgpu-stress-vgpr=64 < %s | FileCheck --check-prefixes=GFX11-PAL %s
; RUN: llc -mtriple=amdgpu11.00--amdpal -amdgpu-use-amdgpu-trackers=1 -verify-misched -amdgpu-stress-vgpr=64 < %s | FileCheck --check-prefixes=GFX11-PAL-GCNTRACKERS %s

; GCN Trackers are sensitive to minor changes in RP, and will avoid scheduling certain instructions, which, if scheduled,
; allow scheduling of other instructions which reduce RP

; CHECK-LABEL: {{^}}return_72xi32:
; GFX11-PAL:    NumSgprs: 33
; GFX11-PAL-GCNTRACKERS:    NumSgprs: 33
; GFX11-PAL:    NumVgprs: 64
; GFX11-PAL-GCNTRACKERS:    NumVgprs: 64
; GFX11-PAL:    ScratchSize: 220
; GFX11-PAL-GCNTRACKERS:    ScratchSize: 248


; CHECK-LABEL: {{^}}call_72xi32:
; GFX11-PAL:    NumSgprs: 40
; GFX11-PAL-GCNTRACKERS:    NumSgprs: 37
; GFX11-PAL:    NumVgprs: 64
; GFX11-PAL-GCNTRACKERS:    NumVgprs: 64
; GFX11-PAL:    ScratchSize: 2780
; GFX11-PAL-GCNTRACKERS:    ScratchSize: 2808


define amdgpu_gfx <72 x i32> @return_72xi32(<72 x i32> %val) #1 {
  ret <72 x i32> %val
}

define amdgpu_gfx void @call_72xi32() #1 {
entry:
  %ret.0 = call amdgpu_gfx <72 x i32> @return_72xi32(<72 x i32> zeroinitializer)
  %val.0 = insertelement <72 x i32> %ret.0, i32 42, i32 0
  %val.1 = insertelement <72 x i32> %val.0, i32 24, i32 58
  %ret.1 = call amdgpu_gfx <72 x i32> @return_72xi32(<72 x i32> %val.1)
  ret void
}

attributes #1 = { nounwind }
