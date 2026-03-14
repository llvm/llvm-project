; RUN: llc < %s -mtriple=i686-- -mattr=+sse2
; PR2566

@0 = external dso_local global i16		; <ptr>:0 [#uses=1]
@1 = external dso_local global <4 x i16>		; <ptr>:1 [#uses=1]

declare void @abort()

define void @t() nounwind {
	load i16, ptr @0		; <i16>:1 [#uses=1]
	zext i16 %1 to i64		; <i64>:2 [#uses=1]
	bitcast i64 %2 to <4 x i16>		; <<4 x i16>>:3 [#uses=1]
	shufflevector <4 x i16> %3, <4 x i16> undef, <4 x i32> zeroinitializer		; <<4 x i16>>:4 [#uses=1]
	store <4 x i16> %4, ptr @1
	ret void
}
