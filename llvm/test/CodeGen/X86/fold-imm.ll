; RUN: llc < %s -march=x86 | FileCheck %s

define i32 @test(i32 %X) nounwind {
entry:
	%0 = add i32 %X, 1
	ret i32 %0
}

; CHECK: test
; CHECK: inc
; CHECK: ret

define i32 @test2(i32 %X) nounwind {
entry:
	%0 = add i32 %X, 4
	ret i32 %0
}

; CHECK: test2
; CHECK: {{add.*4.*$}}
; CHECK: ret
