; RUN: llc < %s -mtriple=xcore -asm-verbose=0 | FileCheck %s
define i32 @ashr(i32 %a, i32 %b) nounwind {
	%1 = ashr i32 %a, %b
	ret i32 %1
}
; CHECK-LABEL: ashr:
; CHECK-NEXT: ashr r0, r0, r1

define i32 @ashri1(i32 %a) nounwind {
	%1 = ashr i32 %a, 24
	ret i32 %1
}
; CHECK-LABEL: ashri1:
; CHECK-NEXT: ashr r0, r0, 24

define i32 @ashri2(i32 %a) nounwind {
	%1 = ashr i32 %a, 31
	ret i32 %1
}
; CHECK-LABEL: ashri2:
; CHECK-NEXT: ashr r0, r0, 32

define i32 @f1(i32 %a) nounwind nounwind {
        %1 = icmp slt i32 %a, 0
	br i1 %1, label %less, label %not_less
less:
	ret i32 10
not_less:
	ret i32 17
}
; CHECK-LABEL: f1:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0

define i32 @f2(i32 %a) nounwind {
        %1 = icmp sge i32 %a, 0
	br i1 %1, label %greater, label %not_greater
greater:
	ret i32 10
not_greater:
	ret i32 17
}
; CHECK-LABEL: f2:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0

define i32 @f3(i32 %a) nounwind {
        %1 = icmp slt i32 %a, 0
	%2 = select i1 %1, i32 10, i32 17
	ret i32 %2
}
; CHECK-LABEL: f3:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0
; CHECK-NEXT: ldc r0, 17
; CHECK: ldc r0, 10

define i32 @f4(i32 %a) nounwind {
        %1 = icmp sge i32 %a, 0
	%2 = select i1 %1, i32 10, i32 17
	ret i32 %2
}
; CHECK-LABEL: f4:
; CHECK-NEXT: ashr r0, r0, 32
; CHECK-NEXT: bt r0
; CHECK-NEXT: ldc r0, 10
; CHECK: ldc r0, 17

define i32 @f5(i32 %a) nounwind {
        %1 = icmp sge i32 %a, 0
	%2 = zext i1 %1 to i32
	ret i32 %2
}
; CHECK-LABEL: f5:
; CHECK-NEXT: not r0, r0
; CHECK-NEXT: mkmsk r1, 5
; CHECK-NEXT: shr r0, r0, r1
