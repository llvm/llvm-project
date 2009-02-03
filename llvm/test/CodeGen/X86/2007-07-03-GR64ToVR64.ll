; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx | grep {movd	%rsi, %mm0}
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx | grep {movd	%rdi, %mm1}
; RUN: llvm-as < %s | llc -mtriple=x86_64-apple-darwin -mattr=+mmx | grep {paddusw	%mm0, %mm1}

@R = external global <1 x i64>		; <<1 x i64>*> [#uses=1]

define void @foo(<1 x i64> %A, <1 x i64> %B) nounwind {
entry:
	%tmp4 = bitcast <1 x i64> %B to <4 x i16>		; <<4 x i16>> [#uses=1]
	%tmp6 = bitcast <1 x i64> %A to <4 x i16>		; <<4 x i16>> [#uses=1]
	%tmp7 = tail call <4 x i16> @llvm.x86.mmx.paddus.w( <4 x i16> %tmp6, <4 x i16> %tmp4 )		; <<4 x i16>> [#uses=1]
	%tmp8 = bitcast <4 x i16> %tmp7 to <1 x i64>		; <<1 x i64>> [#uses=1]
	store <1 x i64> %tmp8, <1 x i64>* @R
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare <4 x i16> @llvm.x86.mmx.paddus.w(<4 x i16>, <4 x i16>)

declare void @llvm.x86.mmx.emms()
