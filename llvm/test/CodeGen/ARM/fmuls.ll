; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test(float %a, float %b) {
entry:
	%0 = fmul float %a, %b
	ret float %0
}

; VFP2: test:
; VFP2: 	vmul.f32	s

; NFP1: test:
; NFP1: 	vmul.f32	d
; NFP0: test:
; NFP0: 	vmul.f32	s

; CORTEXA8: test:
; CORTEXA8: 	vmul.f32	d
; CORTEXA9: test:
; CORTEXA9: 	vmul.f32	s{{.}}, s{{.}}, s{{.}}

; VFP2: test2
define float @test2(float %a) nounwind {
; CHECK-NOT: mul
; CHECK: mov pc, lr
  %ret = fmul float %a, 1.0
  ret float %ret
}

