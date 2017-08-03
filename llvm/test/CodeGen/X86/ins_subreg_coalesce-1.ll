; RUN: llc < %s -march=x86 -mattr=-bmi | FileCheck %s

define fastcc i32 @t() nounwind  {
entry:
; CHECK-LABEL: t:
; CHECK: movzwl 0, %eax
; CHECK: orl $2, %eax
; CHECK: movw %ax, 0
; CHECK: shrl $3, %eax
; CHECK: andl $1, %eax
	br i1 false, label %UnifiedReturnBlock, label %bb4
bb4:		; preds = %entry
	br i1 false, label %bb17, label %bb22
bb17:		; preds = %bb4
	ret i32 1
bb22:		; preds = %bb4
	br i1 true, label %walkExprTree.exit, label %bb4.i
bb4.i:		; preds = %bb22
	ret i32 0
walkExprTree.exit:		; preds = %bb22
	%tmp83 = load i16, i16* null, align 4		; <i16> [#uses=1]
	%tmp84 = or i16 %tmp83, 2		; <i16> [#uses=2]
	store i16 %tmp84, i16* null, align 4
	%tmp98993 = zext i16 %tmp84 to i32		; <i32> [#uses=1]
	%tmp1004 = lshr i32 %tmp98993, 3		; <i32> [#uses=1]
	%tmp100.lobit5 = and i32 %tmp1004, 1		; <i32> [#uses=1]
	ret i32 %tmp100.lobit5
UnifiedReturnBlock:		; preds = %entry
	ret i32 0
}
