; RUN: llvm-as < %s | llc -march=x86 | not grep and

define i32 @t1(i8* %X, i32 %i) {
entry:
	%tmp2 = shl i32 %i, 2		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp2, 1020		; <i32> [#uses=1]
	%tmp7 = getelementptr i8* %X, i32 %tmp4		; <i8*> [#uses=1]
	%tmp78 = bitcast i8* %tmp7 to i32*		; <i32*> [#uses=1]
	%tmp9 = load i32* %tmp78, align 4		; <i32> [#uses=1]
	ret i32 %tmp9
}

define i32 @t2(i16* %X, i32 %i) {
entry:
	%tmp2 = shl i32 %i, 1		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp2, 131070		; <i32> [#uses=1]
	%tmp7 = getelementptr i16* %X, i32 %tmp4		; <i16*> [#uses=1]
	%tmp78 = bitcast i16* %tmp7 to i32*		; <i32*> [#uses=1]
	%tmp9 = load i32* %tmp78, align 4		; <i32> [#uses=1]
	ret i32 %tmp9
}
