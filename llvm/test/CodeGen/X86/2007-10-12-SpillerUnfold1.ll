; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep addss | not grep esp

define fastcc void @fht(float* %fz, i16 signext  %n) {
entry:
	br i1 true, label %bb171.preheader, label %bb431

bb171.preheader:		; preds = %entry
	%tmp176 = add float 0.000000e+00, 1.000000e+00		; <float> [#uses=2]
	%gi.1 = getelementptr float* %fz, i32 0		; <float*> [#uses=2]
	%tmp240 = load float* %gi.1, align 4		; <float> [#uses=1]
	%tmp242 = sub float %tmp240, 0.000000e+00		; <float> [#uses=2]
	%tmp251 = getelementptr float* %fz, i32 0		; <float*> [#uses=1]
	%tmp252 = load float* %tmp251, align 4		; <float> [#uses=1]
	%tmp258 = getelementptr float* %fz, i32 0		; <float*> [#uses=2]
	%tmp259 = load float* %tmp258, align 4		; <float> [#uses=2]
	%tmp261 = mul float %tmp259, %tmp176		; <float> [#uses=1]
	%tmp262 = sub float 0.000000e+00, %tmp261		; <float> [#uses=2]
	%tmp269 = mul float %tmp252, %tmp176		; <float> [#uses=1]
	%tmp276 = mul float %tmp259, 0.000000e+00		; <float> [#uses=1]
	%tmp277 = add float %tmp269, %tmp276		; <float> [#uses=2]
	%tmp281 = getelementptr float* %fz, i32 0		; <float*> [#uses=1]
	%tmp282 = load float* %tmp281, align 4		; <float> [#uses=2]
	%tmp284 = sub float %tmp282, %tmp277		; <float> [#uses=1]
	%tmp291 = add float %tmp282, %tmp277		; <float> [#uses=1]
	%tmp298 = sub float 0.000000e+00, %tmp262		; <float> [#uses=1]
	%tmp305 = add float 0.000000e+00, %tmp262		; <float> [#uses=1]
	%tmp315 = mul float 0.000000e+00, %tmp291		; <float> [#uses=1]
	%tmp318 = mul float 0.000000e+00, %tmp298		; <float> [#uses=1]
	%tmp319 = add float %tmp315, %tmp318		; <float> [#uses=1]
	%tmp329 = add float 0.000000e+00, %tmp319		; <float> [#uses=1]
	store float %tmp329, float* null, align 4
	%tmp336 = sub float %tmp242, 0.000000e+00		; <float> [#uses=1]
	store float %tmp336, float* %tmp258, align 4
	%tmp343 = add float %tmp242, 0.000000e+00		; <float> [#uses=1]
	store float %tmp343, float* null, align 4
	%tmp355 = mul float 0.000000e+00, %tmp305		; <float> [#uses=1]
	%tmp358 = mul float 0.000000e+00, %tmp284		; <float> [#uses=1]
	%tmp359 = add float %tmp355, %tmp358		; <float> [#uses=1]
	%tmp369 = add float 0.000000e+00, %tmp359		; <float> [#uses=1]
	store float %tmp369, float* %gi.1, align 4
	ret void

bb431:		; preds = %entry
	ret void
}
