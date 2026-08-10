; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - | FileCheck %s -check-prefix=VFP2
; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s -check-prefix=NEON
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s -check-prefix=A8

define float @t1(float %acc, float %a, float %b) {
entry:
; VFP2-LABEL: t1:
; VFP2: vmls.f32

; NEON-LABEL: t1:
; NEON: vmls.f32

; A8-LABEL: t1:
; A8: vmul.f32
; A8: vsub.f32
	%0 = fmul float %a, %b
        %1 = fsub float %acc, %0
	ret float %1
}

define double @t2(double %acc, double %a, double %b) {
entry:
; VFP2-LABEL: t2:
; VFP2: vmls.f64

; NEON-LABEL: t2:
; NEON: vmls.f64

; A8-LABEL: t2:
; A8: vmul.f64
; A8: vsub.f64
	%0 = fmul double %a, %b
        %1 = fsub double %acc, %0
	ret double %1
}
