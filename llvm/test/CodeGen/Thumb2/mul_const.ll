; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s
; rdar://7069502

define i32 @t1(i32 %v) nounwind readnone {
entry:
; CHECK: t1:
; CHECK: add.w r0, r0, r0, lsl #3
	%0 = mul i32 %v, 9
	ret i32 %0
}

define i32 @t2(i32 %v) nounwind readnone {
entry:
; CHECK: t2:
; CHECK: rsb r0, r0, r0, lsl #3
	%0 = mul i32 %v, 7
	ret i32 %0
}
