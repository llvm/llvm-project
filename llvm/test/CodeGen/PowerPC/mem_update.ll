; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep addi
; RUN: llc -verify-machineinstrs -code-model=small < %s -mtriple=ppc64-- | \
; RUN:   not grep addi

@Glob = global i64 4

define i32* @test0(i32* %X, i32* %dest) nounwind {
	%Y = getelementptr i32, i32* %X, i32 4
	%A = load i32, i32* %Y
	store i32 %A, i32* %dest
	ret i32* %Y
}

define i32* @test1(i32* %X, i32* %dest) nounwind {
	%Y = getelementptr i32, i32* %X, i32 4
	%A = load i32, i32* %Y
	store i32 %A, i32* %dest
	ret i32* %Y
}

define i16* @test2(i16* %X, i32* %dest) nounwind {
	%Y = getelementptr i16, i16* %X, i32 4
	%A = load i16, i16* %Y
	%B = sext i16 %A to i32
	store i32 %B, i32* %dest
	ret i16* %Y
}

define i16* @test3(i16* %X, i32* %dest) nounwind {
	%Y = getelementptr i16, i16* %X, i32 4
	%A = load i16, i16* %Y
	%B = zext i16 %A to i32
	store i32 %B, i32* %dest
	ret i16* %Y
}

define i16* @test3a(i16* %X, i64* %dest) nounwind {
	%Y = getelementptr i16, i16* %X, i32 4
	%A = load i16, i16* %Y
	%B = sext i16 %A to i64
	store i64 %B, i64* %dest
	ret i16* %Y
}

define i64* @test4(i64* %X, i64* %dest) nounwind {
	%Y = getelementptr i64, i64* %X, i32 4
	%A = load i64, i64* %Y
	store i64 %A, i64* %dest
	ret i64* %Y
}

define i16* @test5(i16* %X) nounwind {
	%Y = getelementptr i16, i16* %X, i32 4
	store i16 7, i16* %Y
	ret i16* %Y
}

define i64* @test6(i64* %X, i64 %A) nounwind {
	%Y = getelementptr i64, i64* %X, i32 4
	store i64 %A, i64* %Y
	ret i64* %Y
}

define i64* @test7(i64* %X, i64 %A) nounwind {
	store i64 %A, i64* @Glob
	ret i64* @Glob
}
