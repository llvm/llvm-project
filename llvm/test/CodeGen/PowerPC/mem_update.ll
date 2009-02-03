; RUN: llvm-as < %s | llc -march=ppc32 -enable-ppc-preinc | \
; RUN:   not grep addi
; RUN: llvm-as < %s | llc -march=ppc64 -enable-ppc-preinc | \
; RUN:   not grep addi

@Glob = global i64 4		; <i64*> [#uses=2]

define i32* @test0(i32* %X, i32* %dest) {
	%Y = getelementptr i32* %X, i32 4		; <i32*> [#uses=2]
	%A = load i32* %Y		; <i32> [#uses=1]
	store i32 %A, i32* %dest
	ret i32* %Y
}

define i32* @test1(i32* %X, i32* %dest) {
	%Y = getelementptr i32* %X, i32 4		; <i32*> [#uses=2]
	%A = load i32* %Y		; <i32> [#uses=1]
	store i32 %A, i32* %dest
	ret i32* %Y
}

define i16* @test2(i16* %X, i32* %dest) {
	%Y = getelementptr i16* %X, i32 4		; <i16*> [#uses=2]
	%A = load i16* %Y		; <i16> [#uses=1]
	%B = sext i16 %A to i32		; <i32> [#uses=1]
	store i32 %B, i32* %dest
	ret i16* %Y
}

define i16* @test3(i16* %X, i32* %dest) {
	%Y = getelementptr i16* %X, i32 4		; <i16*> [#uses=2]
	%A = load i16* %Y		; <i16> [#uses=1]
	%B = zext i16 %A to i32		; <i32> [#uses=1]
	store i32 %B, i32* %dest
	ret i16* %Y
}

define i16* @test3a(i16* %X, i64* %dest) {
	%Y = getelementptr i16* %X, i32 4		; <i16*> [#uses=2]
	%A = load i16* %Y		; <i16> [#uses=1]
	%B = sext i16 %A to i64		; <i64> [#uses=1]
	store i64 %B, i64* %dest
	ret i16* %Y
}

define i64* @test4(i64* %X, i64* %dest) {
	%Y = getelementptr i64* %X, i32 4		; <i64*> [#uses=2]
	%A = load i64* %Y		; <i64> [#uses=1]
	store i64 %A, i64* %dest
	ret i64* %Y
}

define i16* @test5(i16* %X) {
	%Y = getelementptr i16* %X, i32 4		; <i16*> [#uses=2]
	store i16 7, i16* %Y
	ret i16* %Y
}

define i64* @test6(i64* %X, i64 %A) {
	%Y = getelementptr i64* %X, i32 4		; <i64*> [#uses=2]
	store i64 %A, i64* %Y
	ret i64* %Y
}

define i64* @test7(i64* %X, i64 %A) {
	store i64 %A, i64* @Glob
	ret i64* @Glob
}
