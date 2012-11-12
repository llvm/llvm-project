; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s -check-prefix=VFP2
; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s -check-prefix=NFP0
; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s -check-prefix=CORTEXA8
; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s -check-prefix=CORTEXA9

define float @test1(float* %a) {
entry:
	%0 = load float* %a, align 4		; <float> [#uses=2]
	%1 = fsub float -0.000000e+00, %0		; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}
; VFP2: test1:
; VFP2: 	vneg.f32	s{{.*}}, s{{.*}}

; NFP1: test1:
; NFP1: 	vneg.f32	d{{.*}}, d{{.*}}

; NFP0: test1:
; NFP0: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8: test1:
; CORTEXA8: 	vneg.f32	d{{.*}}, d{{.*}}

; CORTEXA9: test1:
; CORTEXA9: 	vneg.f32	s{{.*}}, s{{.*}}

define float @test2(float* %a) {
entry:
	%0 = load float* %a, align 4		; <float> [#uses=2]
	%1 = fmul float -1.000000e+00, %0		; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}
; VFP2: test2:
; VFP2: 	vneg.f32	s{{.*}}, s{{.*}}

; NFP1: test2:
; NFP1: 	vneg.f32	d{{.*}}, d{{.*}}

; NFP0: test2:
; NFP0: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8: test2:
; CORTEXA8: 	vneg.f32	d{{.*}}, d{{.*}}

; CORTEXA9: test2:
; CORTEXA9: 	vneg.f32	s{{.*}}, s{{.*}}

