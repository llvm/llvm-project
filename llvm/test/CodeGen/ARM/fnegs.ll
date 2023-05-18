; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o - \
; RUN:  | FileCheck %s -check-prefix=VFP2

; RUN: llc -mtriple=arm-eabi -mattr=+neon,-neonfp %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP0

; RUN: llc -mtriple=arm-eabi -mattr=+neon,+neonfp %s -o - \
; RUN:  | FileCheck %s -check-prefix=NFP1

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - \
; RUN:  | FileCheck %s -check-prefix=CORTEXA8

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 --enable-unsafe-fp-math %s -o - \
; RUN:  | FileCheck %s -check-prefix=CORTEXA8U

; RUN: llc -mtriple=arm-darwin -mcpu=cortex-a8 %s -o - \
; RUN:  | FileCheck %s -check-prefix=CORTEXA8U

; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 %s -o - \
; RUN:  | FileCheck %s -check-prefix=CORTEXA9

define float @test1(ptr %a) {
entry:
	%0 = load float, ptr %a, align 4		; <float> [#uses=2]
	%1 = fsub float -0.000000e+00, %0		; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}
; VFP2-LABEL: test1:
; VFP2: 	vneg.f32	s{{.*}}, s{{.*}}

; NFP1-LABEL: test1:
; NFP1: 	vneg.f32	d{{.*}}, d{{.*}}

; NFP0-LABEL: test1:
; NFP0: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8-LABEL: test1:
; CORTEXA8: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8U-LABEL: test1:
; CORTEXA8U: 	vneg.f32	d{{.*}}, d{{.*}}

; CORTEXA9-LABEL: test1:
; CORTEXA9: 	vneg.f32	s{{.*}}, s{{.*}}

define float @test2(ptr %a) {
entry:
	%0 = load float, ptr %a, align 4		; <float> [#uses=2]
	%1 = fneg float %0                  ; <float> [#uses=2]
	%2 = fpext float %1 to double		; <double> [#uses=1]
	%3 = fcmp olt double %2, 1.234000e+00		; <i1> [#uses=1]
	%retval = select i1 %3, float %1, float %0		; <float> [#uses=1]
	ret float %retval
}
; VFP2-LABEL: test2:
; VFP2: 	vneg.f32	s{{.*}}, s{{.*}}

; NFP1-LABEL: test2:
; NFP1: 	vneg.f32	d{{.*}}, d{{.*}}

; NFP0-LABEL: test2:
; NFP0: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8-LABEL: test2:
; CORTEXA8: 	vneg.f32	s{{.*}}, s{{.*}}

; CORTEXA8U-LABEL: test2:
; CORTEXA8U: 	vneg.f32	d{{.*}}, d{{.*}}

; CORTEXA9-LABEL: test2:
; CORTEXA9: 	vneg.f32	s{{.*}}, s{{.*}}

; If we're bitcasting an integer to an FP vector, we should avoid the FP/vector unit entirely.
; Make sure that we're flipping the sign bit and only the sign bit of each float (PR20354).
; So instead of something like this:
;    vmov     d16, r0, r1
;    vneg.f32 d16, d16
;    vmov     r0, r1, d16
;
; We should generate:
;    eor     r0, r0, #-214783648
;    eor     r1, r1, #-214783648

define <2 x float> @fneg_bitcast(i64 %i) {
  %bitcast = bitcast i64 %i to <2 x float>
  %fneg = fsub <2 x float> <float -0.0, float -0.0>, %bitcast
  ret <2 x float> %fneg
}
; VFP2-LABEL: fneg_bitcast:
; VFP2-DAG: eor r0, r0, #-2147483648
; VFP2-DAG: eor r1, r1, #-2147483648
; VFP2-NOT:  vneg.f32

; NFP1-LABEL: fneg_bitcast:
; NFP1-DAG: eor r0, r0, #-2147483648
; NFP1-DAG: eor r1, r1, #-2147483648
; NFP1-NOT: vneg.f32

; NFP0-LABEL: fneg_bitcast:
; NFP0-DAG: eor r0, r0, #-2147483648
; NFP0-DAG: eor r1, r1, #-2147483648
; NFP0-NOT: vneg.f32

; CORTEXA8-LABEL: fneg_bitcast:
; CORTEXA8-DAG: eor r0, r0, #-2147483648
; CORTEXA8-DAG: eor r1, r1, #-2147483648
; CORTEXA8-NOT:         vneg.f32

; CORTEXA8U-LABEL: fneg_bitcast:
; CORTEXA8U-DAG: eor r0, r0, #-2147483648
; CORTEXA8U-DAG: eor r1, r1, #-2147483648
; CORTEXA8U-NOT:        vneg.f32

; CORTEXA9-LABEL: fneg_bitcast:
; CORTEXA9-DAG: eor r0, r0, #-2147483648
; CORTEXA9-DAG: eor r1, r1, #-2147483648
; CORTEXA9-NOT:         vneg.f32

