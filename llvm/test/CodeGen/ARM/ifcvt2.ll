; RUN: llc < %s -march=arm -mattr=+v4t | FileCheck %s

define i32 @t1(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK: t1:
; CHECK: bxlt lr
	%tmp2 = icmp sgt i32 %c, 10
	%tmp5 = icmp slt i32 %d, 4
	%tmp8 = or i1 %tmp5, %tmp2
	%tmp13 = add i32 %b, %a
	br i1 %tmp8, label %cond_true, label %UnifiedReturnBlock

cond_true:
	%tmp15 = add i32 %tmp13, %c
	%tmp1821 = sub i32 %tmp15, %d
	ret i32 %tmp1821

UnifiedReturnBlock:
	ret i32 %tmp13
}

define i32 @t2(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK: t2:
; CHECK: bxgt lr
; CHECK: cmp
; CHECK: addge
; CHECK: subge
; CHECK-NOT: bxge lr
; CHECK: bx lr
	%tmp2 = icmp sgt i32 %c, 10
	%tmp5 = icmp slt i32 %d, 4
	%tmp8 = and i1 %tmp5, %tmp2
	%tmp13 = add i32 %b, %a
	br i1 %tmp8, label %cond_true, label %UnifiedReturnBlock

cond_true:
	%tmp15 = add i32 %tmp13, %c
	%tmp1821 = sub i32 %tmp15, %d
	ret i32 %tmp1821

UnifiedReturnBlock:
	ret i32 %tmp13
}
