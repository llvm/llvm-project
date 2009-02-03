; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep {call.*ldexp} | count 4
; rdar://5852514

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define double @t1(i32 %x) nounwind  {
entry:
	%tmp12 = sitofp i32 %x to double		; <double> [#uses=1]
	%exp2 = tail call double @exp2( double %tmp12 )		; <double> [#uses=1]
	ret double %exp2
}

define float @t4(i8 zeroext  %x) nounwind  {
entry:
	%tmp12 = uitofp i8 %x to float		; <float> [#uses=1]
	%tmp3 = tail call float @exp2f( float %tmp12 ) nounwind readonly 		; <float> [#uses=1]
	ret float %tmp3
}

declare float @exp2f(float) nounwind readonly 

define double @t3(i16 zeroext  %x) nounwind  {
entry:
	%tmp12 = uitofp i16 %x to double		; <double> [#uses=1]
	%exp2 = tail call double @exp2( double %tmp12 )		; <double> [#uses=1]
	ret double %exp2
}

define double @t2(i16 signext  %x) nounwind  {
entry:
	%tmp12 = sitofp i16 %x to double		; <double> [#uses=1]
	%exp2 = tail call double @exp2( double %tmp12 )		; <double> [#uses=1]
	ret double %exp2
}

declare double @exp2(double)

