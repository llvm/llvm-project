; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>/dev/null
; PR2433

define i32 @main1(i32 %argc, i8** %argv) nounwind  {
entry:
	br i1 false, label %bb10, label %bb23

bb10:		; preds = %bb10, %entry
	%accum.03 = phi i64 [ %tmp14, %bb10 ], [ 0, %entry ]		; <i64> [#uses=1]
	%i.02 = phi i32 [ %tmp16, %bb10 ], [ 0, %entry ]		; <i32> [#uses=1]
	%d.1.01 = phi i64 [ %tmp5.i, %bb10 ], [ 0, %entry ]		; <i64> [#uses=1]
	%tmp5.i = add i64 %d.1.01, 1		; <i64> [#uses=2]
	%tmp14 = add i64 %accum.03, %tmp5.i		; <i64> [#uses=2]
	%tmp16 = add i32 %i.02, 1		; <i32> [#uses=2]
	%tmp20 = icmp slt i32 %tmp16, 0		; <i1> [#uses=1]
	br i1 %tmp20, label %bb10, label %bb23

bb23:		; preds = %bb10, %entry
	%accum.0.lcssa = phi i64 [ 0, %entry ], [ %tmp14, %bb10 ]		; <i64> [#uses=0]
	ret i32 0
}

define i32 @main2(i32 %argc, i8** %argv) {
entry:
	%tmp8 = tail call i32 @atoi( i8* null ) nounwind readonly 		; <i32> [#uses=1]
	br i1 false, label %bb9, label %bb21

bb9:		; preds = %bb9, %entry
	%accum.03 = phi i64 [ %tmp12, %bb9 ], [ 0, %entry ]		; <i64> [#uses=1]
	%i.02 = phi i32 [ %tmp14, %bb9 ], [ 0, %entry ]		; <i32> [#uses=1]
	%d.1.01 = phi i64 [ %tmp4.i, %bb9 ], [ 0, %entry ]		; <i64> [#uses=1]
	%tmp4.i = add i64 %d.1.01, 1		; <i64> [#uses=2]
	%tmp12 = add i64 %accum.03, %tmp4.i		; <i64> [#uses=2]
	%tmp14 = add i32 %i.02, 1		; <i32> [#uses=2]
	%tmp18 = icmp slt i32 %tmp14, %tmp8		; <i1> [#uses=1]
	br i1 %tmp18, label %bb9, label %bb21

bb21:		; preds = %bb9, %entry
	%accum.0.lcssa = phi i64 [ 0, %entry ], [ %tmp12, %bb9 ]		; <i64> [#uses=0]
	ret i32 0
}

declare i32 @atoi(i8*) nounwind readonly 
