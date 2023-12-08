; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR1533

@array = weak global [101 x i32] zeroinitializer, align 32		; <ptr> [#uses=1]

; CHECK: Loop %bb: backedge-taken count is 100

define void @loop(i32 %x) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%i.01.0 = phi i32 [ 100, %entry ], [ %tmp4, %bb ]		; <i32> [#uses=2]
	%tmp1 = getelementptr [101 x i32], ptr @array, i32 0, i32 %i.01.0		; <ptr> [#uses=1]
	store i32 %x, ptr %tmp1
	%tmp4 = add i32 %i.01.0, -1		; <i32> [#uses=2]
	%tmp7 = icmp sgt i32 %tmp4, -1		; <i1> [#uses=1]
	br i1 %tmp7, label %bb, label %return

return:		; preds = %bb
	ret void
}
