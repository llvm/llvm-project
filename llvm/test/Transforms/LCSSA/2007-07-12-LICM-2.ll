; RUN: opt < %s -passes='loop-mssa(loop-rotate,licm)',simple-loop-unswitch -disable-output
define i32 @main(i32 %argc, ptr %argv) {
entry:
	br label %bb7

bb7:		; preds = %bb7, %entry
	%tmp39 = load <4 x float>, ptr null		; <<4 x float>> [#uses=1]
	%tmp40 = fadd <4 x float> %tmp39, < float 2.000000e+00, float 3.000000e+00, float 1.000000e+00, float 0.000000e+00 >		; <<4 x float>> [#uses=1]
	%tmp43 = fadd <4 x float> %tmp40, < float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 2.000000e+00 >		; <<4 x float>> [#uses=1]
	%tmp46 = fadd <4 x float> %tmp43, < float 3.000000e+00, float 0.000000e+00, float 2.000000e+00, float 4.000000e+00 >		; <<4 x float>> [#uses=1]
	%tmp49 = fadd <4 x float> %tmp46, < float 0.000000e+00, float 4.000000e+00, float 6.000000e+00, float 1.000000e+00 >		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp49, ptr null
	br i1 false, label %bb7, label %bb56

bb56:		; preds = %bb7
	ret i32 0
}
