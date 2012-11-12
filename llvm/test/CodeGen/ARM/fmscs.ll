; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NEON
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=A8

define float @t1(float %acc, float %a, float %b) {
entry:
; VFP2: t1:
; VFP2: vnmls.f32

; NEON: t1:
; NEON: vnmls.f32

; A8: t1:
; A8: vmul.f32
; A8: vsub.f32
	%0 = fmul float %a, %b
        %1 = fsub float %0, %acc
	ret float %1
}

define double @t2(double %acc, double %a, double %b) {
entry:
; VFP2: t2:
; VFP2: vnmls.f64

; NEON: t2:
; NEON: vnmls.f64

; A8: t2:
; A8: vmul.f64
; A8: vsub.f64
	%0 = fmul double %a, %b
        %1 = fsub double %0, %acc
	ret double %1
}
