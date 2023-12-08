; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - \
; RUN:  | FileCheck %s -check-prefix=VFP2

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP1

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 --enable-unsafe-fp-math %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP1U

; RUN: llc -mtriple=arm-darwin -mcpu=cortex-a8 %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP1U

; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP0

define float @test(float %a, float %b) {
entry:
	%0 = fsub float %a, %b
	ret float %0
}

; VFP2: vsub.f32	s
; NFP1U: vsub.f32	d
; NFP1: vsub.f32	s
; NFP0: vsub.f32	s
