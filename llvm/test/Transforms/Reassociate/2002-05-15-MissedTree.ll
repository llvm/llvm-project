; RUN: opt < %s -reassociate -instcombine -S | FileCheck %s

define i32 @test1(i32 %A, i32 %B) {
; CHECK-LABEL: test1
; CHECK: %Z = add i32 %B, %A
; CHECK: ret i32 %Z
	%W = add i32 %B, -5
	%Y = add i32 %A, 5
	%Z = add i32 %W, %Y
	ret i32 %Z
}
