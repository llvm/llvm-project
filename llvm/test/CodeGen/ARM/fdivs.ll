; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - | FileCheck %s -check-prefix=VFP2
; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s -check-prefix=NFP0
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
	%0 = fdiv float %a, %b
	ret float %0
}

; VFP2-LABEL: test:
; VFP2: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}

; NFP1-LABEL: test:
; NFP1: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}
; NFP0-LABEL: test:
; NFP0: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}

; CORTEXA8-LABEL: test:
; CORTEXA8: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}
; CORTEXA9-LABEL: test:
; CORTEXA9: 	vdiv.f32	s{{.}}, s{{.}}, s{{.}}
