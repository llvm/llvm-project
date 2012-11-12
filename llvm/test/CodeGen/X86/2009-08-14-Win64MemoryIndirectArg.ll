; RUN: llc < %s
target triple = "x86_64-mingw"

; ModuleID = 'mm.bc'
	%0 = type opaque		; type %0
	%1 = type opaque		; type %1

define internal fastcc float @computeMipmappingRho(%0* %shaderExecutionStatePtr, i32 %index, <4 x float> %texCoord, <4 x float> %texCoordDX, <4 x float> %texCoordDY) readonly {
indexCheckBlock:
	%indexCmp = icmp ugt i32 %index, 16		; <i1> [#uses=1]
	br i1 %indexCmp, label %zeroReturnBlock, label %primitiveTextureFetchBlock

primitiveTextureFetchBlock:		; preds = %indexCheckBlock
	%pointerArithmeticTmp = bitcast %0* %shaderExecutionStatePtr to i8*		; <i8*> [#uses=1]
	%pointerArithmeticTmp1 = getelementptr i8* %pointerArithmeticTmp, i64 1808		; <i8*> [#uses=1]
	%pointerArithmeticTmp2 = bitcast i8* %pointerArithmeticTmp1 to %1**		; <%1**> [#uses=1]
	%primitivePtr = load %1** %pointerArithmeticTmp2		; <%1*> [#uses=1]
	%pointerArithmeticTmp3 = bitcast %1* %primitivePtr to i8*		; <i8*> [#uses=1]
	%pointerArithmeticTmp4 = getelementptr i8* %pointerArithmeticTmp3, i64 19408		; <i8*> [#uses=1]
	%pointerArithmeticTmp5 = bitcast i8* %pointerArithmeticTmp4 to %1**		; <%1**> [#uses=1]
	%primitiveTexturePtr = getelementptr %1** %pointerArithmeticTmp5, i32 %index		; <%1**> [#uses=1]
	%primitiveTexturePtr6 = load %1** %primitiveTexturePtr		; <%1*> [#uses=2]
	br label %textureCheckBlock

textureCheckBlock:		; preds = %primitiveTextureFetchBlock
	%texturePtrInt = ptrtoint %1* %primitiveTexturePtr6 to i64		; <i64> [#uses=1]
	%testTextureNULL = icmp eq i64 %texturePtrInt, 0		; <i1> [#uses=1]
	br i1 %testTextureNULL, label %zeroReturnBlock, label %rhoCalculateBlock

rhoCalculateBlock:		; preds = %textureCheckBlock
	%pointerArithmeticTmp7 = bitcast %1* %primitiveTexturePtr6 to i8*		; <i8*> [#uses=1]
	%pointerArithmeticTmp8 = getelementptr i8* %pointerArithmeticTmp7, i64 640		; <i8*> [#uses=1]
	%pointerArithmeticTmp9 = bitcast i8* %pointerArithmeticTmp8 to <4 x float>*		; <<4 x float>*> [#uses=1]
	%dimensionsPtr = load <4 x float>* %pointerArithmeticTmp9, align 1		; <<4 x float>> [#uses=2]
	%texDiffDX = fsub <4 x float> %texCoordDX, %texCoord		; <<4 x float>> [#uses=1]
	%texDiffDY = fsub <4 x float> %texCoordDY, %texCoord		; <<4 x float>> [#uses=1]
	%ddx = fmul <4 x float> %texDiffDX, %dimensionsPtr		; <<4 x float>> [#uses=2]
	%ddx10 = fmul <4 x float> %texDiffDY, %dimensionsPtr		; <<4 x float>> [#uses=2]
	%ddxSquared = fmul <4 x float> %ddx, %ddx		; <<4 x float>> [#uses=3]
	%0 = shufflevector <4 x float> %ddxSquared, <4 x float> %ddxSquared, <4 x i32> <i32 1, i32 0, i32 0, i32 0>		; <<4 x float>> [#uses=1]
	%dxSquared = fadd <4 x float> %ddxSquared, %0		; <<4 x float>> [#uses=1]
	%1 = call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %dxSquared)		; <<4 x float>> [#uses=1]
	%ddySquared = fmul <4 x float> %ddx10, %ddx10		; <<4 x float>> [#uses=3]
	%2 = shufflevector <4 x float> %ddySquared, <4 x float> %ddySquared, <4 x i32> <i32 1, i32 0, i32 0, i32 0>		; <<4 x float>> [#uses=1]
	%dySquared = fadd <4 x float> %ddySquared, %2		; <<4 x float>> [#uses=1]
	%3 = call <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float> %dySquared)		; <<4 x float>> [#uses=1]
	%4 = call <4 x float> @llvm.x86.sse.max.ss(<4 x float> %1, <4 x float> %3)		; <<4 x float>> [#uses=1]
	%rho = extractelement <4 x float> %4, i32 0		; <float> [#uses=1]
	ret float %rho

zeroReturnBlock:		; preds = %textureCheckBlock, %indexCheckBlock
	ret float 0.000000e+00
}

declare <4 x float> @llvm.x86.sse.sqrt.ss(<4 x float>) nounwind readnone

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>) nounwind readnone
