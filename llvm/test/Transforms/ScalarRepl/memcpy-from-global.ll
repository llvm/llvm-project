; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | not grep {call.*memcpy}
@C.0.1248 = internal constant [128 x float] [ float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float -1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float -1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float -1.000000e+00, float 0.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 ], align 32		; <[128 x float]*> [#uses=1]

define float @grad4(i32 %hash, float %x, float %y, float %z, float %w) {
entry:
	%lookupTable = alloca [128 x float], align 16		; <[128 x float]*> [#uses=5]
	%lookupTable1 = bitcast [128 x float]* %lookupTable to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %lookupTable1, i8* bitcast ([128 x float]* @C.0.1248 to i8*), i32 512, i32 16 )
	%tmp3 = shl i32 %hash, 2		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp3, 124		; <i32> [#uses=4]
	%tmp753 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp5		; <float*> [#uses=1]
	%tmp9 = load float* %tmp753		; <float> [#uses=1]
	%tmp11 = mul float %tmp9, %x		; <float> [#uses=1]
	%tmp13 = add float %tmp11, 0.000000e+00		; <float> [#uses=1]
	%tmp17.sum52 = or i32 %tmp5, 1		; <i32> [#uses=1]
	%tmp1851 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp17.sum52		; <float*> [#uses=1]
	%tmp19 = load float* %tmp1851		; <float> [#uses=1]
	%tmp21 = mul float %tmp19, %y		; <float> [#uses=1]
	%tmp23 = add float %tmp21, %tmp13		; <float> [#uses=1]
	%tmp27.sum50 = or i32 %tmp5, 2		; <i32> [#uses=1]
	%tmp2849 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp27.sum50		; <float*> [#uses=1]
	%tmp29 = load float* %tmp2849		; <float> [#uses=1]
	%tmp31 = mul float %tmp29, %z		; <float> [#uses=1]
	%tmp33 = add float %tmp31, %tmp23		; <float> [#uses=1]
	%tmp37.sum48 = or i32 %tmp5, 3		; <i32> [#uses=1]
	%tmp3847 = getelementptr [128 x float]* %lookupTable, i32 0, i32 %tmp37.sum48		; <float*> [#uses=1]
	%tmp39 = load float* %tmp3847		; <float> [#uses=1]
	%tmp41 = mul float %tmp39, %w		; <float> [#uses=1]
	%tmp43 = add float %tmp41, %tmp33		; <float> [#uses=1]
	ret float %tmp43
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32)
