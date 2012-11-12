; Loop Simplify should turn phi nodes like X = phi [X, Y]  into just Y, eliminating them.
; RUN: opt < %s -loop-simplify -S | grep phi | count 6

@A = weak global [3000000 x i32] zeroinitializer		; <[3000000 x i32]*> [#uses=1]
@B = weak global [20000 x i32] zeroinitializer		; <[20000 x i32]*> [#uses=1]
@C = weak global [100 x i32] zeroinitializer		; <[100 x i32]*> [#uses=1]
@Z = weak global i32 0		; <i32*> [#uses=2]

define i32 @main() {
entry:
	tail call void @__main( )
	br label %loopentry.1
loopentry.1:		; preds = %loopexit.1, %entry
	%indvar20 = phi i32 [ 0, %entry ], [ %indvar.next21, %loopexit.1 ]		; <i32> [#uses=1]
	%a.1 = phi i32* [ getelementptr ([3000000 x i32]* @A, i32 0, i32 0), %entry ], [ %inc.0, %loopexit.1 ]		; <i32*> [#uses=1]
	br label %no_exit.2
no_exit.2:		; preds = %loopexit.2, %no_exit.2, %loopentry.1
	%a.0.4.ph = phi i32* [ %a.1, %loopentry.1 ], [ %inc.0, %loopexit.2 ], [ %a.0.4.ph, %no_exit.2 ]		; <i32*> [#uses=3]
	%b.1.4.ph = phi i32* [ getelementptr ([20000 x i32]* @B, i32 0, i32 0), %loopentry.1 ], [ %inc.1, %loopexit.2 ], [ %b.1.4.ph, %no_exit.2 ]		; <i32*> [#uses=3]
	%indvar17 = phi i32 [ 0, %loopentry.1 ], [ %indvar.next18, %loopexit.2 ], [ %indvar17, %no_exit.2 ]		; <i32> [#uses=2]
	%indvar = phi i32 [ %indvar.next, %no_exit.2 ], [ 0, %loopexit.2 ], [ 0, %loopentry.1 ]		; <i32> [#uses=5]
	%b.1.4.rec = bitcast i32 %indvar to i32		; <i32> [#uses=1]
	%gep.upgrd.1 = zext i32 %indvar to i64		; <i64> [#uses=1]
	%c.2.4 = getelementptr [100 x i32]* @C, i32 0, i64 %gep.upgrd.1		; <i32*> [#uses=1]
	%gep.upgrd.2 = zext i32 %indvar to i64		; <i64> [#uses=1]
	%a.0.4 = getelementptr i32* %a.0.4.ph, i64 %gep.upgrd.2		; <i32*> [#uses=1]
	%gep.upgrd.3 = zext i32 %indvar to i64		; <i64> [#uses=1]
	%b.1.4 = getelementptr i32* %b.1.4.ph, i64 %gep.upgrd.3		; <i32*> [#uses=1]
	%inc.0.rec = add i32 %b.1.4.rec, 1		; <i32> [#uses=2]
	%inc.0 = getelementptr i32* %a.0.4.ph, i32 %inc.0.rec		; <i32*> [#uses=2]
	%tmp.13 = load i32* %a.0.4		; <i32> [#uses=1]
	%inc.1 = getelementptr i32* %b.1.4.ph, i32 %inc.0.rec		; <i32*> [#uses=1]
	%tmp.15 = load i32* %b.1.4		; <i32> [#uses=1]
	%tmp.18 = load i32* %c.2.4		; <i32> [#uses=1]
	%tmp.16 = mul i32 %tmp.15, %tmp.13		; <i32> [#uses=1]
	%tmp.19 = mul i32 %tmp.16, %tmp.18		; <i32> [#uses=1]
	%tmp.20 = load i32* @Z		; <i32> [#uses=1]
	%tmp.21 = add i32 %tmp.19, %tmp.20		; <i32> [#uses=1]
	store i32 %tmp.21, i32* @Z
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, 100		; <i1> [#uses=1]
	br i1 %exitcond, label %loopexit.2, label %no_exit.2
loopexit.2:		; preds = %no_exit.2
	%indvar.next18 = add i32 %indvar17, 1		; <i32> [#uses=2]
	%exitcond19 = icmp eq i32 %indvar.next18, 200		; <i1> [#uses=1]
	br i1 %exitcond19, label %loopexit.1, label %no_exit.2
loopexit.1:		; preds = %loopexit.2
	%indvar.next21 = add i32 %indvar20, 1		; <i32> [#uses=2]
	%exitcond22 = icmp eq i32 %indvar.next21, 300		; <i1> [#uses=1]
	br i1 %exitcond22, label %return, label %loopentry.1
return:		; preds = %loopexit.1
	ret i32 undef
}

declare void @__main()
