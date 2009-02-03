; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis > %t
; RUN: not grep {call.*floor(} %t
; RUN: grep {call.*floorf(} %t
; RUN: not grep {call.*ceil(} %t
; RUN: grep {call.*ceilf(} %t
; RUN: not grep {call.*nearbyint(} %t
; RUN: grep {call.*nearbyintf(} %t
; XFAIL: sparc

declare double @floor(double)

declare double @ceil(double)

declare double @nearbyint(double)

define float @test_floor(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
        ; --> floorf
	%E = call double @floor( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

define float @test_ceil(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
	; --> ceilf
        %E = call double @ceil( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

define float @test_nearbyint(float %C) {
	%D = fpext float %C to double		; <double> [#uses=1]
	; --> nearbyintf
        %E = call double @nearbyint( double %D )		; <double> [#uses=1]
	%F = fptrunc double %E to float		; <float> [#uses=1]
	ret float %F
}

