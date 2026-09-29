; RUN: llc -mtriple=lanai < %s -o - | FileCheck %s

define void @f1() {
	%c = alloca i8, align 1
	ret void
}
; CHECK-LABEL: f1:
; CHECK: sub %sp, 0x10

define i32 @f2() {
	ret i32 1
}
; CHECK-LABEL: f2:
; CHECK: sub %sp, 0x8
