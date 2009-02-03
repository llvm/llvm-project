; RUN: bugpoint %s  -bugpoint-crashcalls -silence-passes

; Test to make sure that arguments are removed from the function if they are 
; unnecessary.

declare i32 @test2()

define i32 @test(i32 %A, i32 %B, float %C) {
	call i32 @test2()
	ret i32 %1
}
