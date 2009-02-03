; RUN: llvm-as < %s | opt -jump-threading -disable-output
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9"
@Link = global [1 x i32] [ i32 -1 ]		; <[1 x i32]*> [#uses=2]
@W = global [1 x i32] [ i32 2 ]		; <[1 x i32]*> [#uses=1]

define i32 @f(i32 %k, i32 %p) nounwind  {
entry:
	br label %bb

bb:		; preds = %bb56, %bb76.loopexit.us, %entry
	%j.2 = phi i32 [ 0, %entry ], [ 1, %bb56 ], [ 1, %bb76.loopexit.us ]		; <i32> [#uses=5]
	%pdest.2 = phi i32 [ 0, %entry ], [ %pdest.8, %bb56 ], [ %pdest.7.us, %bb76.loopexit.us ]		; <i32> [#uses=3]
	%p_addr.0 = phi i32 [ %p, %entry ], [ 0, %bb56 ], [ %p_addr.1.us, %bb76.loopexit.us ]		; <i32> [#uses=3]
	%k_addr.0 = phi i32 [ %k, %entry ], [ %tmp59, %bb56 ], [ %tmp59.us, %bb76.loopexit.us ]		; <i32> [#uses=4]
	%tmp2 = icmp sgt i32 %pdest.2, 2		; <i1> [#uses=1]
	br i1 %tmp2, label %bb4.preheader, label %bb13

bb4.preheader:		; preds = %bb
	%tmp109 = sub i32 1, %j.2		; <i32> [#uses=2]
	%tmp110 = icmp slt i32 %tmp109, -2		; <i1> [#uses=1]
	%smax111 = select i1 %tmp110, i32 -2, i32 %tmp109		; <i32> [#uses=2]
	%tmp112 = add i32 %j.2, %smax111		; <i32> [#uses=2]
	br label %bb4

bb4:		; preds = %bb4, %bb4.preheader
	%indvar = phi i32 [ 0, %bb4.preheader ], [ %indvar.next, %bb4 ]		; <i32> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %tmp112		; <i1> [#uses=1]
	br i1 %exitcond, label %bb13.loopexit, label %bb4

bb13.loopexit:		; preds = %bb4
	%tmp = add i32 %j.2, %pdest.2		; <i32> [#uses=1]
	%tmp102 = add i32 %tmp, %smax111		; <i32> [#uses=1]
	%tmp104 = add i32 %tmp112, -1		; <i32> [#uses=1]
	%tmp106 = sub i32 %j.2, %tmp104		; <i32> [#uses=1]
	%tmp107 = add i32 %tmp106, -1		; <i32> [#uses=1]
	br label %bb13

bb13:		; preds = %bb13.loopexit, %bb
	%j.1 = phi i32 [ %tmp107, %bb13.loopexit ], [ %j.2, %bb ]		; <i32> [#uses=4]
	%pdest.1 = phi i32 [ %tmp102, %bb13.loopexit ], [ %pdest.2, %bb ]		; <i32> [#uses=2]
	%tmp15 = icmp eq i32 %j.1, 1		; <i1> [#uses=1]
	br i1 %tmp15, label %bb82, label %bb27.preheader

bb27.preheader:		; preds = %bb13
	%tmp21 = icmp eq i32 %j.1, %p_addr.0		; <i1> [#uses=0]
	br label %bb27.outer

bb27.outer:		; preds = %bb27.outer.bb24.split_crit_edge, %bb27.preheader
	%indvar118 = phi i32 [ 0, %bb27.preheader ], [ %indvar.next119, %bb27.outer.bb24.split_crit_edge ]		; <i32> [#uses=2]
	%pdest.3.ph = add i32 %indvar118, %pdest.1		; <i32> [#uses=2]
	%tmp30 = icmp sgt i32 %pdest.3.ph, %p_addr.0		; <i1> [#uses=1]
	br i1 %tmp30, label %bb27.outer.bb24.split_crit_edge, label %bb27.outer.split

bb27.outer.bb24.split_crit_edge:		; preds = %bb27.outer
	%indvar.next119 = add i32 %indvar118, 1		; <i32> [#uses=1]
	br label %bb27.outer

bb27.outer.split:		; preds = %bb27.outer
	%tmp35 = getelementptr [1 x i32]* @W, i32 0, i32 %k_addr.0		; <i32*> [#uses=3]
	%tmp48 = icmp slt i32 %p_addr.0, 1		; <i1> [#uses=1]
	%tmp53 = icmp sgt i32 %k_addr.0, 0		; <i1> [#uses=1]
	br label %bb33

bb33:		; preds = %bb51.split, %bb27.outer.split
	%pdest.5 = phi i32 [ %pdest.3.ph, %bb27.outer.split ], [ %pdest.4, %bb51.split ]		; <i32> [#uses=1]
	%tmp36 = load i32* %tmp35, align 4		; <i32> [#uses=2]
	br i1 %tmp48, label %bb37.us, label %bb37

bb37.us:		; preds = %bb42.us, %bb37.us, %bb33
	%D1361.1.us = phi i32 [ %tmp36, %bb33 ], [ 0, %bb42.us ], [ %D1361.1.us, %bb37.us ]		; <i32> [#uses=2]
	%tmp39.us = icmp eq i32 %D1361.1.us, 0		; <i1> [#uses=1]
	br i1 %tmp39.us, label %bb37.us, label %bb42.us

bb42.us:		; preds = %bb37.us
	store i32 0, i32* %tmp35, align 4
	br label %bb37.us

bb37:		; preds = %bb33
	%tmp39 = icmp eq i32 %tmp36, 0		; <i1> [#uses=1]
	br i1 %tmp39, label %bb51.split, label %bb42

bb42:		; preds = %bb37
	store i32 0, i32* %tmp35, align 4
	br label %bb51.split

bb51.split:		; preds = %bb42, %bb37
	%pdest.4 = phi i32 [ 1, %bb42 ], [ %pdest.5, %bb37 ]		; <i32> [#uses=3]
	br i1 %tmp53, label %bb33, label %bb56.preheader

bb56.preheader:		; preds = %bb51.split
	%tmp7394 = icmp sgt i32 %j.1, 0		; <i1> [#uses=1]
	br i1 %tmp7394, label %bb56.us, label %bb56

bb56.us:		; preds = %bb76.loopexit.us, %bb56.preheader
	%pdest.8.us = phi i32 [ %pdest.4, %bb56.preheader ], [ %pdest.7.us, %bb76.loopexit.us ]		; <i32> [#uses=1]
	%k_addr.1.us = phi i32 [ %k_addr.0, %bb56.preheader ], [ %tmp59.us, %bb76.loopexit.us ]		; <i32> [#uses=1]
	%tmp58.us = getelementptr [1 x i32]* @Link, i32 0, i32 %k_addr.1.us		; <i32*> [#uses=1]
	%tmp59.us = load i32* %tmp58.us, align 4		; <i32> [#uses=3]
	%tmp6295.us = icmp ne i32 %tmp59.us, -1		; <i1> [#uses=2]
	br label %bb60.us

bb60.us:		; preds = %bb60.us, %bb56.us
	%pdest.7.reg2mem.0.us = phi i32 [ %pdest.8.us, %bb56.us ], [ %pdest.7.us, %bb60.us ]		; <i32> [#uses=1]
	%p_addr.1.reg2mem.0.us = phi i32 [ 0, %bb56.us ], [ %p_addr.1.us, %bb60.us ]		; <i32> [#uses=1]
	%tmp67.us = zext i1 %tmp6295.us to i32		; <i32> [#uses=2]
	%pdest.7.us = add i32 %pdest.7.reg2mem.0.us, %tmp67.us		; <i32> [#uses=3]
	%p_addr.1.us = add i32 %p_addr.1.reg2mem.0.us, %tmp67.us		; <i32> [#uses=3]
	%tmp73.us = icmp slt i32 %p_addr.1.us, %j.1		; <i1> [#uses=1]
	br i1 %tmp73.us, label %bb60.us, label %bb76.loopexit.us

bb76.loopexit.us:		; preds = %bb60.us
	br i1 %tmp6295.us, label %bb56.us, label %bb

bb56:		; preds = %bb56, %bb56.preheader
	%pdest.8 = phi i32 [ %pdest.4, %bb56.preheader ], [ %pdest.8, %bb56 ]		; <i32> [#uses=2]
	%k_addr.1 = phi i32 [ %k_addr.0, %bb56.preheader ], [ %tmp59, %bb56 ]		; <i32> [#uses=1]
	%tmp58 = getelementptr [1 x i32]* @Link, i32 0, i32 %k_addr.1		; <i32*> [#uses=1]
	%tmp59 = load i32* %tmp58, align 4		; <i32> [#uses=3]
	%tmp6295 = icmp ne i32 %tmp59, -1		; <i1> [#uses=1]
	br i1 %tmp6295, label %bb56, label %bb

bb82:		; preds = %bb13
	ret i32 %pdest.1
}

define i32 @main() nounwind  {
entry:
	%tmp1 = tail call i32 @f( i32 0, i32 2 ) nounwind 		; <i32> [#uses=1]
	%tmp2 = icmp eq i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %bb, label %bb4

bb:		; preds = %entry
	tail call void @abort( ) noreturn nounwind 
	unreachable

bb4:		; preds = %entry
	ret i32 0
}

declare void @abort() noreturn nounwind 
