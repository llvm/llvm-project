; RUN: llc -mtriple=xcore < %s | FileCheck %s

; Only needs one ladd
define i64 @f1(i32 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %x to i64		; <i64> [#uses=1]
	%1 = zext i32 %y to i64		; <i64> [#uses=1]
	%2 = add i64 %1, %0		; <i64> [#uses=1]
	ret i64 %2
}
; CHECK-LABEL: f1:
; CHECK: ldc r2, 0
; CHECK-NEXT: ladd r1, r0, r1, r0, r2
; CHECK-NEXT: retsp 0

; Only needs one lsub and one neg
define i64 @f2(i32 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %x to i64		; <i64> [#uses=1]
	%1 = zext i32 %y to i64		; <i64> [#uses=1]
	%2 = sub i64 %1, %0		; <i64> [#uses=1]
	ret i64 %2
}
; CHECK-LABEL: f2:
; CHECK: ldc r2, 0
; CHECK-NEXT: lsub r1, r0, r1, r0, r2
; CHECK-NEXT: neg r1, r1
; CHECK-NEXT: retsp 0

; Should compile to one ladd and one add
define i64 @f3(i64 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %y to i64		; <i64> [#uses=1]
	%1 = add i64 %x, %0		; <i64> [#uses=1]
	ret i64 %1
}
; CHECK-LABEL: f3:
; CHECK: ldc r3, 0
; CHECK-NEXT: ladd r2, r0, r0, r2, r3
; CHECK-NEXT: add r1, r1, r2
; CHECK-NEXT: retsp 0

; Should compile to one ladd and one add
define i64 @f4(i32 %x, i64 %y) nounwind {
entry:
	%0 = zext i32 %x to i64		; <i64> [#uses=1]
	%1 = add i64 %0, %y		; <i64> [#uses=1]
	ret i64 %1
}
; CHECK-LABEL: f4:
; CHECK: ldc r3, 0
; CHECK-NEXT: ladd r1, r0, r0, r1, r3
; CHECK-NEXT: add r1, r2, r1
; CHECK-NEXT: retsp 0

; Should compile to one lsub and one sub
define i64 @f5(i64 %x, i32 %y) nounwind {
entry:
	%0 = zext i32 %y to i64		; <i64> [#uses=1]
	%1 = sub i64 %x, %0		; <i64> [#uses=1]
	ret i64 %1
}
; CHECK-LABEL: f5:
; CHECK: ldc r3, 0
; CHECK-NEXT: lsub r2, r0, r0, r2, r3
; CHECK-NEXT: sub r1, r1, r2
; CHECK-NEXT: retsp 0
