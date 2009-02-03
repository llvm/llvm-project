; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep "phi double" | count 1

define void @foobar(i32 %n) nounwind {
entry:
	icmp eq i32 %n, 0		; <i1>:0 [#uses=2]
	br i1 %0, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%umax = select i1 %0, i32 1, i32 %n		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	tail call void @bar( i32 %i.03 ) nounwind
	uitofp i32 %i.03 to double		; <double>:1 [#uses=1]
	tail call void @foo( double %1 ) nounwind
	%indvar.next = add i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast because the mantissa bits for double are not enough
; to hold all of i64 IV bits.
define void @foobar2(i64 %n) nounwind {
entry:
	icmp eq i64 %n, 0		; <i1>:0 [#uses=2]
	br i1 %0, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%umax = select i1 %0, i64 1, i64 %n		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i64> [#uses=3]
	trunc i64 %i.03 to i32		; <i32>:1 [#uses=1]
	tail call void @bar( i32 %1 ) nounwind
	uitofp i64 %i.03 to double		; <double>:2 [#uses=1]
	tail call void @foo( double %2 ) nounwind
	%indvar.next = add i64 %i.03, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast due to potentional overflow.
define void @foobar3() nounwind {
entry:
	tail call i32 (...)* @nn( ) nounwind		; <i32>:0 [#uses=1]
	icmp eq i32 %0, 0		; <i1>:1 [#uses=1]
	br i1 %1, label %return, label %bb

bb:		; preds = %bb, %entry
	%i.03 = phi i32 [ 0, %entry ], [ %3, %bb ]		; <i32> [#uses=3]
	tail call void @bar( i32 %i.03 ) nounwind
	uitofp i32 %i.03 to double		; <double>:2 [#uses=1]
	tail call void @foo( double %2 ) nounwind
	add i32 %i.03, 1		; <i32>:3 [#uses=2]
	tail call i32 (...)* @nn( ) nounwind		; <i32>:4 [#uses=1]
	icmp ugt i32 %4, %3		; <i1>:5 [#uses=1]
	br i1 %5, label %bb, label %return

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast due to overflow.
define void @foobar4() nounwind {
entry:
	br label %bb.nph

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i8 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	%tmp2 = sext i8 %i.03 to i32		; <i32>:0 [#uses=1]
	tail call void @bar( i32 %tmp2 ) nounwind
	%tmp3 = uitofp i8 %i.03 to double		; <double>:1 [#uses=1]
	tail call void @foo( double %tmp3 ) nounwind
	%indvar.next = add i8 %i.03, 1		; <i32> [#uses=2]
        %tmp = sext i8 %indvar.next to i32
	%exitcond = icmp eq i32 %tmp, 32767		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

declare void @bar(i32)

declare void @foo(double)

declare i32 @nn(...)

