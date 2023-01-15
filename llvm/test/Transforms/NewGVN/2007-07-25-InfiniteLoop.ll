; RUN: opt < %s -passes=newgvn -S | FileCheck %s

	%struct.INT2 = type { i32, i32 }
@blkshifts = external global ptr		; <ptr> [#uses=2]

define i32 @xcompact() {
entry:
	store ptr null, ptr @blkshifts, align 4
	br label %bb

bb:		; preds = %bb, %entry
	%tmp10 = load ptr, ptr @blkshifts, align 4		; <ptr> [#uses=0]
; CHECK-NOT:  %tmp10
	br label %bb
}
