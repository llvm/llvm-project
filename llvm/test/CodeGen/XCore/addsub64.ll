; RUN: llc < %s -mtriple=xcore | FileCheck %s
define i64 @add64(i64 %a, i64 %b) {
	%result = add i64 %a, %b
	ret i64 %result
}
; CHECK: add64
; CHECK: ldc r11, 0
; CHECK-NEXT: ladd r2, r0, r0, r2, r11
; CHECK-NEXT: ladd r2, r1, r1, r3, r2
; CHECK-NEXT: retsp 0

define i64 @sub64(i64 %a, i64 %b) {
	%result = sub i64 %a, %b
	ret i64 %result
}
; CHECK: sub64
; CHECK: ldc r11, 0
; CHECK-NEXT: lsub r2, r0, r0, r2, r11
; CHECK-NEXT: lsub r2, r1, r1, r3, r2
; CHECK-NEXT: retsp 0

define i64 @maccu(i64 %a, i32 %b, i32 %c) {
entry:
	%0 = zext i32 %b to i64
	%1 = zext i32 %c to i64
	%2 = mul i64 %1, %0
	%3 = add i64 %2, %a
	ret i64 %3
}
; CHECK-LABEL: maccu:
; CHECK: maccu r1, r0, r3, r2
; CHECK-NEXT: retsp 0

define i64 @maccs(i64 %a, i32 %b, i32 %c) {
entry:
	%0 = sext i32 %b to i64
	%1 = sext i32 %c to i64
	%2 = mul i64 %1, %0
	%3 = add i64 %2, %a
	ret i64 %3
}
; CHECK-LABEL: maccs:
; CHECK: maccs r1, r0, r3, r2
; CHECK-NEXT: retsp 0

define i64 @lmul(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
	%0 = zext i32 %a to i64
	%1 = zext i32 %b to i64
	%2 = zext i32 %c to i64
	%3 = zext i32 %d to i64
	%4 = mul i64 %1, %0
	%5 = add i64 %4, %2
	%6 = add i64 %5, %3
	ret i64 %6
}
; CHECK-LABEL: lmul:
; CHECK: lmul r1, r0, r1, r0, r2, r3
; CHECK-NEXT: retsp 0
