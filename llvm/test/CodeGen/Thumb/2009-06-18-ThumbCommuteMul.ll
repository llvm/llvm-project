; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s

define i32 @a(i32 %x, i32 %y) nounwind readnone {
entry:
	%mul = mul i32 %y, %x		; <i32> [#uses=1]
	ret i32 %mul
}

; CHECK: r0

