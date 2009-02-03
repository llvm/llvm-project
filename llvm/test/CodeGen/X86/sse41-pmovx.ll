; RUN: llvm-as < %s | llc -march=x86    -mattr=sse41 | not grep movd
; RUN: llvm-as < %s | llc -march=x86    -mattr=sse41 | not grep movq
; RUN: llvm-as < %s | llc -march=x86    -mattr=sse41 | grep pmovsxbd
; RUN: llvm-as < %s | llc -march=x86    -mattr=sse41 | grep pmovsxwd
; RUN: llvm-as < %s | llc -march=x86    -mattr=sse41 | grep pmovsxbq
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=sse41 -mtriple=x86_64-apple-darwin | grep movq | count 1
; RUN: llvm-as < %s | llc -march=x86-64 -mattr=sse41 -mtriple=x86_64-unknown-linux-gnu | not grep movq

define <2 x i64> @t1(i32* %p) nounwind {
entry:
	%0 = load i32* %p, align 4		; <i32> [#uses=1]
	%1 = insertelement <4 x i32> undef, i32 %0, i32 0		; <<4 x i32>> [#uses=1]
	%2 = insertelement <4 x i32> %1, i32 0, i32 1		; <<4 x i32>> [#uses=1]
	%3 = insertelement <4 x i32> %2, i32 0, i32 2		; <<4 x i32>> [#uses=1]
	%4 = insertelement <4 x i32> %3, i32 0, i32 3		; <<4 x i32>> [#uses=1]
	%5 = bitcast <4 x i32> %4 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%6 = tail call <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8> %5) nounwind readnone		; <<4 x i32>> [#uses=1]
	%7 = bitcast <4 x i32> %6 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %7
}

declare <4 x i32> @llvm.x86.sse41.pmovsxbd(<16 x i8>) nounwind readnone

define <2 x i64> @t2(i64* %p) nounwind readonly {
entry:
	%0 = load i64* %p		; <i64> [#uses=1]
	%tmp2 = insertelement <2 x i64> zeroinitializer, i64 %0, i32 0		; <<2 x i64>> [#uses=1]
	%1 = bitcast <2 x i64> %tmp2 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%2 = tail call <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16> %1) nounwind readnone		; <<4 x i32>> [#uses=1]
	%3 = bitcast <4 x i32> %2 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %3
}

declare <4 x i32> @llvm.x86.sse41.pmovsxwd(<8 x i16>) nounwind readnone


@gv = external global i16		; <i16*> [#uses=1]

define <2 x i64> @t3() nounwind {
entry:
	%0 = load i16* @gv, align 2		; <i16> [#uses=1]
	%1 = insertelement <8 x i16> undef, i16 %0, i32 0		; <<8 x i16>> [#uses=1]
	%2 = bitcast <8 x i16> %1 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%3 = tail call <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8> %2) nounwind readnone		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %3
}

declare <2 x i64> @llvm.x86.sse41.pmovzxbq(<16 x i8>) nounwind readnone
