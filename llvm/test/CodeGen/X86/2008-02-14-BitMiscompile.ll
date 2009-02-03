; RUN: llvm-as < %s | llc -march=x86 | grep and
define i32 @test(i1 %A) {
	%B = zext i1 %A to i32		; <i32> [#uses=1]
	%C = sub i32 0, %B		; <i32> [#uses=1]
	%D = and i32 %C, 255		; <i32> [#uses=1]
	ret i32 %D
}

