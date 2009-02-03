; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlps
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | not grep movups
; rdar://6523650

	%struct.vector4_t = type { <4 x float> }

define void @swizzle(i8* nocapture %a, %struct.vector4_t* nocapture %b, %struct.vector4_t* nocapture %c) nounwind {
entry:
	%0 = getelementptr %struct.vector4_t* %b, i32 0, i32 0		; <<4 x float>*> [#uses=2]
	%1 = load <4 x float>* %0, align 4		; <<4 x float>> [#uses=1]
	%tmp.i = bitcast i8* %a to double*		; <double*> [#uses=1]
	%tmp1.i = load double* %tmp.i		; <double> [#uses=1]
	%2 = insertelement <2 x double> undef, double %tmp1.i, i32 0		; <<2 x double>> [#uses=1]
	%tmp2.i = bitcast <2 x double> %2 to <4 x float>		; <<4 x float>> [#uses=1]
	%3 = shufflevector <4 x float> %1, <4 x float> %tmp2.i, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x float>> [#uses=1]
	store <4 x float> %3, <4 x float>* %0, align 4
	ret void
}
