; RUN: llvm-as < %s | opt -scalarrepl -disable-output

target datalayout = "E-p:32:32"

define i32 @test(i64 %L) {
	%X = alloca i32		; <i32*> [#uses=2]
	%Y = bitcast i32* %X to i64*		; <i64*> [#uses=1]
	store i64 0, i64* %Y
	%Z = load i32* %X		; <i32> [#uses=1]
	ret i32 %Z
}

