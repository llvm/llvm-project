; RUN: llc -mtriple=arm-eabi < %s -o - | FileCheck %s

define void @f1() {
	%c = alloca i8, align 1
	ret void
}
; CHECK-LABEL: f1:
; CHECK: add

define i32 @f2() {
	ret i32 1
}
; CHECK-LABEL: f2:
; CHECK-NOT: add
