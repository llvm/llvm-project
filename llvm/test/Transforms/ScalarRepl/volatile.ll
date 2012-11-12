; RUN: opt < %s -scalarrepl -S | grep "load volatile"
; RUN: opt < %s -scalarrepl -S | grep "store volatile"

define i32 @voltest(i32 %T) {
	%A = alloca {i32, i32}
	%B = getelementptr {i32,i32}* %A, i32 0, i32 0
	store volatile i32 %T, i32* %B

	%C = getelementptr {i32,i32}* %A, i32 0, i32 1
	%X = load volatile i32* %C
	ret i32 %X
}
