; RUN: llc < %s -march=ppc32 -fp-contract=fast -mattr=-vsx -disable-ppc-vsx-fma-mutation=false | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mattr=+vsx -mcpu=pwr7 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-VSX %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -fp-contract=fast -mcpu=pwr8 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-P8 %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -fp-contract=fast -mcpu=pwr8 -disable-ppc-vsx-fma-mutation=false | FileCheck -check-prefix=CHECK-P8 %s

declare double @dummy1(double) #0
declare double @dummy2(double, double) #0
declare double @dummy3(double, double, double) #0
declare float @dummy4(float, float) #0

define double @test_FMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMADD1:
; CHECK: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD1:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMADD2:
; CHECK: fmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMADD2:
; CHECK-VSX: xsmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FMSUB1:
; CHECK: fmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FMSUB1:
; CHECK-VSX: xsmsubmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FMSUB2(double %A, double %B, double %C, double %D) {
	%E = fmul double %A, %B 	; <double> [#uses=2]
	%F = fadd double %E, %C 	; <double> [#uses=1]
	%G = fsub double %E, %D 	; <double> [#uses=1]
	%H = call double @dummy2(double %F, double %G)      ; <double> [#uses=1]
	ret double %H
; CHECK-LABEL: test_FMSUB2:
; CHECK: fmadd
; CHECK-NEXT: fmsub

; CHECK-VSX-LABEL: test_FMSUB2:
; CHECK-VSX: xsmaddadp
; CHECK-VSX-NEXT: xsmsubmdp
}

define double @test_FNMADD1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMADD1:
; CHECK: fnmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMADD1:
; CHECK-VSX: xsnmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FNMADD2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fadd double %C, %D		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMADD2:
; CHECK: fnmadd
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMADD2:
; CHECK-VSX: xsnmaddmdp
; CHECK-VSX-NEXT: blr
}

define double @test_FNMSUB1(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %C, %D		; <double> [#uses=1]
	ret double %E
; CHECK-LABEL: test_FNMSUB1:
; CHECK: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUB1:
; CHECK-VSX: xsnmsubmdp
}

define double @test_FNMSUB2(double %A, double %B, double %C) {
	%D = fmul double %A, %B		; <double> [#uses=1]
	%E = fsub double %D, %C		; <double> [#uses=1]
	%F = fsub double -0.000000e+00, %E		; <double> [#uses=1]
	ret double %F
; CHECK-LABEL: test_FNMSUB2:
; CHECK: fnmsub
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUB2:
; CHECK-VSX: xsnmsubmdp
; CHECK-VSX-NEXT: blr
}

define float @test_FNMSUBS(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
; CHECK-LABEL: test_FNMSUBS:
; CHECK: fnmsubs
; CHECK-NEXT: blr

; CHECK-VSX-LABEL: test_FNMSUBS:
; CHECK-VSX: fnmsubs
; CHECK-VSX-NEXT: blr
}

define float @test_XSMADDMSP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fadd float %C, %D		; <float> [#uses=1]
	ret float %E
; CHECK-P8-LABEL: test_XSMADDMSP:
; CHECK-P8: xsmaddmsp
; CHECK-P8-NEXT: blr
}

define float @test_XSMSUBMSP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	ret float %E
; CHECK-P8-LABEL: test_XSMSUBMSP:
; CHECK-P8: xsmsubmsp
; CHECK-P8-NEXT: blr
}

define float @test_XSMADDASP(float %A, float %B, float %C, float %D) {
	%E = fmul float %A, %B 	; <float> [#uses=2]
	%F = fadd float %E, %C 	; <float> [#uses=1]
	%G = fsub float %E, %D 	; <float> [#uses=1]
	%H = call float @dummy4(float %F, float %G)      ; <float> [#uses=1]
	ret float %H
; CHECK-P8-LABEL: test_XSMADDASP:
; CHECK-P8: xsmaddasp
; CHECK-P8-NEXT: xsmsubmsp
}

define float @test_XSMSUBASP(float %A, float %B, float %C, float %D) {
	%E = fmul float %A, %B 	; <float> [#uses=2]
	%F = fsub float %E, %C 	; <float> [#uses=1]
	%G = fsub float %E, %D 	; <float> [#uses=1]
	%H = call float @dummy4(float %F, float %G)      ; <float> [#uses=1]
	ret float %H
; CHECK-P8-LABEL: test_XSMSUBASP:
; CHECK-P8: xsmsubasp
; CHECK-P8-NEXT: xsmsubmsp
}

define float @test_XSNMADDMSP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fadd float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
; CHECK-P8-LABEL: test_XSNMADDMSP:
; CHECK-P8: xsnmaddmsp
; CHECK-P8-NEXT: blr
}

define float @test_XSNMSUBMSP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	ret float %F
; CHECK-P8-LABEL: test_XSNMSUBMSP:
; CHECK-P8: xsnmsubmsp
; CHECK-P8-NEXT: blr
}

define float @test_XSNMADDASP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fadd float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	%H = call float @dummy4(float %E, float %F)      ; <float> [#uses=1]
	ret float %F
; CHECK-P8-LABEL: test_XSNMADDASP:
; CHECK-P8: xsnmaddasp
}

define float @test_XSNMSUBASP(float %A, float %B, float %C) {
	%D = fmul float %A, %B		; <float> [#uses=1]
	%E = fsub float %D, %C		; <float> [#uses=1]
	%F = fsub float -0.000000e+00, %E		; <float> [#uses=1]
	%H = call float @dummy4(float %E, float %F)      ; <float> [#uses=1]
	ret float %F
; CHECK-P8-LABEL: test_XSNMSUBASP:
; CHECK-P8: xsnmsubasp
}
