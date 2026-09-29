; RUN: llc < %s -mtriple=i686-apple-macosx -mcpu=yonah | FileCheck %s
; CHECK-NOT:     {{((xor|and)ps|movd)}}

; These operations should be done in integer registers, eliminating constant
; pool loads, movd's etc.

define i32 @test1(float %x) nounwind  {
entry:
	%tmp2 = fsub float -0.000000e+00, %x		; <float> [#uses=1]
	%tmp210 = bitcast float %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp210
}

define i32 @test2(float %x) nounwind  {
entry:
	%tmp2 = tail call float @copysignf( float 1.000000e+00, float %x ) nounwind readnone 		; <float> [#uses=1]
	%tmp210 = bitcast float %tmp2 to i32		; <i32> [#uses=1]
	ret i32 %tmp210
}

declare float @copysignf(float, float) nounwind readnone 

