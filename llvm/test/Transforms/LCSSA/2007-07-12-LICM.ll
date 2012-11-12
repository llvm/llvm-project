; RUN: opt < %s -loop-rotate -licm -loop-unswitch -disable-output
define i32 @main(i32 %argc, i8** %argv) {
entry:
	br label %bb7

bb7:		; preds = %bb7, %entry
	%tmp39 = load <4 x float>* null		; <<4 x float>> [#uses=1]
	%tmp40 = fadd <4 x float> %tmp39, < float 2.000000e+00, float 3.000000e+00, float 1.000000e+00, float 0.000000e+00 >		; <<4 x float>> [#uses=0]
	store <4 x float> zeroinitializer, <4 x float>* null
	br i1 false, label %bb7, label %bb56

bb56:		; preds = %bb7
	ret i32 0
}
