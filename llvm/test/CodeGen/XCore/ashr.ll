; RUN: llc < %s -march=xcore -asm-verbose=0 | FileCheck %s
define i32 @ashr(i32 %a, i32 %b) {
	%1 = ashr i32 %a, %b
	ret i32 %1
}
; CHECK: ashr:
; CHECK-NEXT: ashr r0, r0, r1

define i32 @ashri1(i32 %a) {
	%1 = ashr i32 %a, 24
	ret i32 %1
}
; CHECK: ashri1:
; CHECK-NEXT: ashr r0, r0, 24

define i32 @ashri2(i32 %a) {
	%1 = ashr i32 %a, 31
	ret i32 %1
}
; CHECK: ashri2:
; CHECK-NEXT: ashr r0, r0, 32

define i32 @f1(i32 %a) {
        %1 = icmp slt i32 %a, 0
	br i1 %1, label %less, label %not_less
less:
	ret i32 10
not_less:
	ret i32 17
}
; CHECK: f1:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0

define i32 @f2(i32 %a) {
        %1 = icmp sge i32 %a, 0
	br i1 %1, label %greater, label %not_greater
greater:
	ret i32 10
not_greater:
	ret i32 17
}
; CHECK: f2:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0

define i32 @f3(i32 %a) {
        %1 = icmp slt i32 %a, 0
	%2 = select i1 %1, i32 10, i32 17
	ret i32 %2
}
; CHECK: f3:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0
; CHECK-NEXT: ldc r0, 17
; CHECK: ldc r0, 10

define i32 @f4(i32 %a) {
        %1 = icmp sge i32 %a, 0
	%2 = select i1 %1, i32 10, i32 17
	ret i32 %2
}
; CHECK: f4:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0
; CHECK-NEXT: ldc r0, 10
; CHECK: ldc r0, 17

define i32 @f5(i32 %a) {
        %1 = icmp sge i32 %a, 0
	%2 = zext i1 %1 to i32
	ret i32 %2
}
; CHECK: f5:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: eq r0, r0, 0
