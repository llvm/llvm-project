; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=NFP1
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0

define float @test(float %a, float %b) {
entry:
	%0 = fsub float %a, %b
	ret float %0
}

; VFP2: vsub.f32	s
; NFP1: vsub.f32	d
; NFP0: vsub.f32	s
