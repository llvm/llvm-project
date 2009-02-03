; This should be one add and two multiplies.

; RUN: llvm-as < %s | \
; RUN:   opt -reassociate -instcombine | llvm-dis > %t 
; RUN: grep mul %t | count 2
; RUN: grep add %t | count 1

define i32 @test(i32 %A, i32 %B, i32 %C) {
	%aa = mul i32 %A, %A		; <i32> [#uses=1]
	%aab = mul i32 %aa, %B		; <i32> [#uses=1]
	%ac = mul i32 %A, %C		; <i32> [#uses=1]
	%aac = mul i32 %ac, %A		; <i32> [#uses=1]
	%r = add i32 %aab, %aac		; <i32> [#uses=1]
	ret i32 %r
}

