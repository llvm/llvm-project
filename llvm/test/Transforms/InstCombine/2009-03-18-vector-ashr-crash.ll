; RUN: opt < %s -passes=instcombine | llvm-dis
; PR3826

define void @0(ptr, ptr) {
	%3 = alloca ptr		; <ptr> [#uses=1]
	%4 = load <4 x i16>, ptr null, align 1		; <<4 x i16>> [#uses=1]
	%5 = ashr <4 x i16> %4, <i16 5, i16 5, i16 5, i16 5>		; <<4 x i16>> [#uses=1]
	%6 = load ptr, ptr %3		; <ptr> [#uses=1]
	store <4 x i16> %5, ptr %6, align 1
	ret void
}
