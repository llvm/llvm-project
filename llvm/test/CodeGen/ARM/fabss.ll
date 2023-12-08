; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -mtriple=arm-apple-ios -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -mtriple=arm-apple-ios -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
        %dum = fadd float %a, %b
	%0 = tail call float @fabsf(float %dum) readnone
        %dum1 = fadd float %0, %b
	ret float %dum1
}

declare float @fabsf(float)

; VFP2-LABEL: test:
; VFP2: 	vabs.f32	s

; NFP1-LABEL: test:
; NFP1: 	vabs.f32	d
; NFP0-LABEL: test:
; NFP0: 	vabs.f32	s

; CORTEXA8-LABEL: test:
; CORTEXA8:     vadd.f32        [[D1:d[0-9]+]]
; CORTEXA8: 	vabs.f32	{{d[0-9]+}}, [[D1]]

; CORTEXA9-LABEL: test:
; CORTEXA9: 	vabs.f32	s{{.}}, s{{.}}
