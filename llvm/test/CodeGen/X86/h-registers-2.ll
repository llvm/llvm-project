; RUN: llc < %s -mtriple=i686-- | FileCheck %s

; Use an h register, but don't omit the explicit shift for
; non-address use(s).

define i32 @foo(i8* %x, i32 %y) nounwind {
; CHECK-LABEL: foo:
; CHECK-NOT: ret
; CHECK: movzbl %{{[abcd]h}},
; CHECK-NOT: ret
; CHECK: shll $3,
; CHECK: ret

	%t0 = lshr i32 %y, 8		; <i32> [#uses=1]
	%t1 = and i32 %t0, 255		; <i32> [#uses=2]
  %t2 = shl i32 %t1, 3
	%t3 = getelementptr i8, i8* %x, i32 %t2		; <i8*> [#uses=1]
	store i8 77, i8* %t3, align 4
	ret i32 %t2
}
