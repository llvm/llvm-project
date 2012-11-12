; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
	%0 = fadd float %a, %b
	ret float %0
}

; VFP2: test:
; VFP2: 	vadd.f32	s

; NFP1: test:
; NFP1: 	vadd.f32	d
; NFP0: test:
; NFP0: 	vadd.f32	s

; CORTEXA8: test:
; CORTEXA8: 	vadd.f32	d
; CORTEXA9: test:
; CORTEXA9: 	vadd.f32	s{{.}}, s{{.}}, s{{.}}
