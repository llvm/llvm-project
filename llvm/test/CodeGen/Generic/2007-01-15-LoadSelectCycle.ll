; RUN: llc < %s
; PR1114

declare i1 @foo()

define i32 @test(i32* %A, i32* %B) {
	%a = load i32* %A
	%b = load i32* %B
	%cond = call i1 @foo()
	%c = select i1 %cond, i32 %a, i32 %b
	ret i32 %c
}
