; RUN: opt < %s -indvars
; rdar://6817574

define i32 @t1() nounwind ssp {
entry:
	br label %bb32

bb32:		; preds = %bb32, %entry
	%mbPartIdx.0.reg2mem.0 = phi i8 [ %2, %bb32 ], [ 0, %entry ]		; <i8> [#uses=3]
	%0 = and i8 %mbPartIdx.0.reg2mem.0, 1		; <i8> [#uses=0]
	%1 = zext i8 %mbPartIdx.0.reg2mem.0 to i64		; <i64> [#uses=0]
	%2 = add i8 %mbPartIdx.0.reg2mem.0, 1		; <i8> [#uses=2]
	%3 = icmp ugt i8 %2, 3		; <i1> [#uses=1]
	br i1 %3, label %bb41, label %bb32

bb41:		; preds = %bb32
	ret i32 0
}

define i32 @t2() nounwind ssp {
entry:
	br label %bb116

bb116:		; preds = %bb116, %entry
	%mbPartIdx.1.reg2mem.0 = phi i8 [ %3, %bb116 ], [ 0, %entry ]		; <i8> [#uses=3]
	%0 = and i8 %mbPartIdx.1.reg2mem.0, 1		; <i8> [#uses=1]
	%1 = zext i8 %mbPartIdx.1.reg2mem.0 to i64		; <i64> [#uses=0]
	%2 = zext i8 %0 to i32		; <i32> [#uses=0]
	%3 = add i8 %mbPartIdx.1.reg2mem.0, 1		; <i8> [#uses=2]
	%4 = icmp ugt i8 %3, 3		; <i1> [#uses=1]
	br i1 %4, label %bb131, label %bb116

bb131:		; preds = %bb116
	unreachable
}
