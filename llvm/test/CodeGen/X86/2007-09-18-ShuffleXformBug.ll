; RUN: llc < %s -march=x86 -mattr=+sse2 | grep -- -86

define i16 @f(<4 x float>* %tmp116117.i1061.i) nounwind {
entry:
	alloca [4 x <4 x float>]		; <[4 x <4 x float>]*>:0 [#uses=167]
	alloca [4 x <4 x float>]		; <[4 x <4 x float>]*>:1 [#uses=170]
	alloca [4 x <4 x i32>]		; <[4 x <4 x i32>]*>:2 [#uses=12]
	%.sub6235.i = getelementptr [4 x <4 x float>]* %0, i32 0, i32 0		; <<4 x float>*> [#uses=76]
	%.sub.i = getelementptr [4 x <4 x float>]* %1, i32 0, i32 0		; <<4 x float>*> [#uses=59]

	%tmp124.i1062.i = getelementptr <4 x float>* %tmp116117.i1061.i, i32 63		; <<4 x float>*> [#uses=1]
	%tmp125.i1063.i = load <4 x float>* %tmp124.i1062.i		; <<4 x float>> [#uses=5]
	%tmp828.i1077.i = shufflevector <4 x float> %tmp125.i1063.i, <4 x float> undef, <4 x i32> < i32 1, i32 1, i32 1, i32 1 >		; <<4 x float>> [#uses=4]
	%tmp704.i1085.i = load <4 x float>* %.sub6235.i		; <<4 x float>> [#uses=1]
	%tmp712.i1086.i = call <4 x float> @llvm.x86.sse.max.ps( <4 x float> %tmp704.i1085.i, <4 x float> %tmp828.i1077.i )		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp712.i1086.i, <4 x float>* %.sub.i

	%tmp2587.i1145.gep.i = getelementptr [4 x <4 x float>]* %1, i32 0, i32 0, i32 2		; <float*> [#uses=1]
	%tmp5334.i = load float* %tmp2587.i1145.gep.i		; <float> [#uses=5]
	%tmp2723.i1170.i = insertelement <4 x float> undef, float %tmp5334.i, i32 2		; <<4 x float>> [#uses=5]
	store <4 x float> %tmp2723.i1170.i, <4 x float>* %.sub6235.i

	%tmp1406.i1367.i = shufflevector <4 x float> %tmp2723.i1170.i, <4 x float> undef, <4 x i32> < i32 2, i32 2, i32 2, i32 2 >		; <<4 x float>> [#uses=1]
	%tmp84.i1413.i = load <4 x float>* %.sub6235.i		; <<4 x float>> [#uses=1]
	%tmp89.i1415.i = fmul <4 x float> %tmp84.i1413.i, %tmp1406.i1367.i		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp89.i1415.i, <4 x float>* %.sub.i
        ret i16 0
}

declare <4 x float> @llvm.x86.sse.max.ps(<4 x float>, <4 x float>)
