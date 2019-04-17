; RUN: opt < %s -instcombine | llvm-dis
; PR3826

define void @0(<4 x i16>*, <4 x i16>*) {
	%3 = alloca <4 x i16>*		; <<4 x i16>**> [#uses=1]
	%4 = load <4 x i16>, <4 x i16>* null, align 1		; <<4 x i16>> [#uses=1]
	%5 = ashr <4 x i16> %4, <i16 5, i16 5, i16 5, i16 5>		; <<4 x i16>> [#uses=1]
	%6 = load <4 x i16>*, <4 x i16>** %3		; <<4 x i16>*> [#uses=1]
	store <4 x i16> %5, <4 x i16>* %6, align 1
	ret void
}
