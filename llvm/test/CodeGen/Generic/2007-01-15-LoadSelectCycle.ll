; RUN: llc < %s
; PR1114

declare i1 @foo()

define i32 @test(ptr %A, ptr %B) {
	%a = load i32, ptr %A
	%b = load i32, ptr %B
	%cond = call i1 @foo()
	%c = select i1 %cond, i32 %a, i32 %b
	ret i32 %c
}
