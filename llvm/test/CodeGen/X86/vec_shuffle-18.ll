; RUN: llc < %s -march=x86 -mattr=+sse2 -mtriple=i686-apple-darwin8.8.0 | grep mov | count 7

	%struct.vector4_t = type { <4 x float> }

define void @swizzle(i8* %a, %struct.vector4_t* %b, %struct.vector4_t* %c) nounwind  {
entry:
	%tmp9 = getelementptr %struct.vector4_t* %b, i32 0, i32 0		; <<4 x float>*> [#uses=2]
	%tmp10 = load <4 x float>* %tmp9, align 16		; <<4 x float>> [#uses=1]
	%tmp14 = bitcast i8* %a to double*		; <double*> [#uses=1]
	%tmp15 = load double* %tmp14		; <double> [#uses=1]
	%tmp16 = insertelement <2 x double> undef, double %tmp15, i32 0		; <<2 x double>> [#uses=1]
	%tmp18 = bitcast <2 x double> %tmp16 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp19 = shufflevector <4 x float> %tmp10, <4 x float> %tmp18, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp19, <4 x float>* %tmp9, align 16
	%tmp28 = getelementptr %struct.vector4_t* %c, i32 0, i32 0		; <<4 x float>*> [#uses=2]
	%tmp29 = load <4 x float>* %tmp28, align 16		; <<4 x float>> [#uses=1]
	%tmp26 = getelementptr i8* %a, i32 8		; <i8*> [#uses=1]
	%tmp33 = bitcast i8* %tmp26 to double*		; <double*> [#uses=1]
	%tmp34 = load double* %tmp33		; <double> [#uses=1]
	%tmp35 = insertelement <2 x double> undef, double %tmp34, i32 0		; <<2 x double>> [#uses=1]
	%tmp37 = bitcast <2 x double> %tmp35 to <4 x float>		; <<4 x float>> [#uses=1]
	%tmp38 = shufflevector <4 x float> %tmp29, <4 x float> %tmp37, <4 x i32> < i32 4, i32 5, i32 2, i32 3 >		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp38, <4 x float>* %tmp28, align 16
	ret void
}
