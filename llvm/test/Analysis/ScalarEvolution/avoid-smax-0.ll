; RUN: opt < %s -scalar-evolution -analyze | grep "Loop %bb3: backedge-taken count is (-1 + %n)"

; We don't want to use a max in the trip count expression in
; this testcase.

define void @foo(i32 %n, i32* %p, i32* %q) nounwind {
entry:
	icmp sgt i32 %n, 0
	br i1 %0, label %bb, label %return

bb:
	load i32* %q, align 4
	icmp eq i32 %1, 0
	br i1 %2, label %return, label %bb3.preheader

bb3.preheader:
	br label %bb3

bb3:
	%i.0 = phi i32 [ %7, %bb3 ], [ 0, %bb3.preheader ]
	getelementptr i32* %p, i32 %i.0
	load i32* %3, align 4
	add i32 %4, 1
	getelementptr i32* %p, i32 %i.0
	store i32 %5, i32* %6, align 4
	add i32 %i.0, 1
	icmp slt i32 %7, %n
	br i1 %8, label %bb3, label %return.loopexit

return.loopexit:
	br label %return

return:
	ret void
}
