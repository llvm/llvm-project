; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {volatile load}
; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {volatile store}

define i32 @voltest(i32 %T) {
	%A = alloca {i32, i32}
	%B = getelementptr {i32,i32}* %A, i32 0, i32 0
	volatile store i32 %T, i32* %B

	%C = getelementptr {i32,i32}* %A, i32 0, i32 1
	%X = volatile load i32* %C
	ret i32 %X
}
