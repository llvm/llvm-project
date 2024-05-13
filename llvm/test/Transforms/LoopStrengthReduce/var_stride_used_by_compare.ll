; Base should not be i*3, it should be i*2.
; RUN: opt < %s -loop-reduce -S | \
; RUN:   not grep "mul.*%i, 3"

; Indvar should not start at zero:
; RUN: opt < %s -loop-reduce -S | \
; RUN:   not grep "phi i32 .* 0"
; END.

; mul uint %i, 3

target datalayout = "e-p:32:32-n32"
target triple = "i686-apple-darwin8"
@flags2 = external global [8193 x i8], align 32		; <ptr> [#uses=1]

define void @foo(i32 %k, i32 %i.s) {
entry:
	%i = bitcast i32 %i.s to i32		; <i32> [#uses=2]
	%k_addr.012 = shl i32 %i.s, 1		; <i32> [#uses=1]
	%tmp14 = icmp sgt i32 %k_addr.012, 8192		; <i1> [#uses=1]
	br i1 %tmp14, label %return, label %bb.preheader
bb.preheader:		; preds = %entry
	%tmp. = shl i32 %i, 1		; <i32> [#uses=1]
	br label %bb
bb:		; preds = %bb, %bb.preheader
	%indvar = phi i32 [ %indvar.next, %bb ], [ 0, %bb.preheader ]		; <i32> [#uses=2]
	%tmp.15 = mul i32 %indvar, %i		; <i32> [#uses=1]
	%tmp.16 = add i32 %tmp.15, %tmp.		; <i32> [#uses=2]
	%k_addr.0.0 = bitcast i32 %tmp.16 to i32		; <i32> [#uses=1]
	%gep.upgrd.1 = zext i32 %tmp.16 to i64		; <i64> [#uses=1]
	%tmp = getelementptr [8193 x i8], ptr @flags2, i32 0, i64 %gep.upgrd.1		; <ptr> [#uses=1]
	store i8 0, ptr %tmp
	%k_addr.0 = add i32 %k_addr.0.0, %i.s		; <i32> [#uses=1]
	%tmp.upgrd.2 = icmp sgt i32 %k_addr.0, 8192		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp.upgrd.2, label %return.loopexit, label %bb
return.loopexit:		; preds = %bb
	br label %return
return:		; preds = %return.loopexit, %entry
	ret void
}
