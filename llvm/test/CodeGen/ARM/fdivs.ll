; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
	%0 = fdiv float %a, %b
	ret float %0
}

; VFP2: test:
; VFP2: 	vdiv.f32	s0, s2, s0

; NFP1: test:
; NFP1: 	vdiv.f32	s0, s2, s0
; NFP0: test:
; NFP0: 	vdiv.f32	s0, s2, s0

; CORTEXA8: test:
; CORTEXA8: 	vdiv.f32	s0, s2, s0
; CORTEXA9: test:
; CORTEXA9: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}
