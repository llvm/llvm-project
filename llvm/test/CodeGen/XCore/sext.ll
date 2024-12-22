; RUN: llc < %s -march=xcore | FileCheck %s
define i32 @sext1(i32 %a) {
	%1 = trunc i32 %a to i1
	%2 = sext i1 %1 to i32
	ret i32 %2
}
; CHECK-LABEL: sext1:
; CHECK: sext r0, 1

define i32 @sext2(i32 %a) {
	%1 = trunc i32 %a to i2
	%2 = sext i2 %1 to i32
	ret i32 %2
}
; CHECK-LABEL: sext2:
; CHECK: sext r0, 2

define i32 @sext8(i32 %a) {
	%1 = trunc i32 %a to i8
	%2 = sext i8 %1 to i32
	ret i32 %2
}
; CHECK-LABEL: sext8:
; CHECK: sext r0, 8

define i32 @sext16(i32 %a) {
	%1 = trunc i32 %a to i16
	%2 = sext i16 %1 to i32
	ret i32 %2
}
; CHECK-LABEL: sext16:
; CHECK: sext r0, 16
