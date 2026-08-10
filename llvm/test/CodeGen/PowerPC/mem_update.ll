; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep addi
; RUN: llc -verify-machineinstrs -code-model=small < %s -mtriple=ppc64-- | \
; RUN:   not grep addi

@Glob = global i64 4

define ptr @test0(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i32, ptr %X, i32 4
	%A = load i32, ptr %Y
	store i32 %A, ptr %dest
	ret ptr %Y
}

define ptr @test1(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i32, ptr %X, i32 4
	%A = load i32, ptr %Y
	store i32 %A, ptr %dest
	ret ptr %Y
}

define ptr @test2(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i16, ptr %X, i32 4
	%A = load i16, ptr %Y
	%B = sext i16 %A to i32
	store i32 %B, ptr %dest
	ret ptr %Y
}

define ptr @test3(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i16, ptr %X, i32 4
	%A = load i16, ptr %Y
	%B = zext i16 %A to i32
	store i32 %B, ptr %dest
	ret ptr %Y
}

define ptr @test3a(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i16, ptr %X, i32 4
	%A = load i16, ptr %Y
	%B = sext i16 %A to i64
	store i64 %B, ptr %dest
	ret ptr %Y
}

define ptr @test4(ptr %X, ptr %dest) nounwind {
	%Y = getelementptr i64, ptr %X, i32 4
	%A = load i64, ptr %Y
	store i64 %A, ptr %dest
	ret ptr %Y
}

define ptr @test5(ptr %X) nounwind {
	%Y = getelementptr i16, ptr %X, i32 4
	store i16 7, ptr %Y
	ret ptr %Y
}

define ptr @test6(ptr %X, i64 %A) nounwind {
	%Y = getelementptr i64, ptr %X, i32 4
	store i64 %A, ptr %Y
	ret ptr %Y
}

define ptr @test7(ptr %X, i64 %A) nounwind {
	store i64 %A, ptr @Glob
	ret ptr @Glob
}
