; RUN: opt < %s -loop-reduce -S -mtriple=x86_64-unknown-unknown 2>&1 | FileCheck %s

; Provide legal integer types.
target datalayout = "n8:16:32:64"


define void @foobar(i32 %n) nounwind {

; CHECK-LABEL:  foobar(
; CHECK:        phi double

entry:
	%cond = icmp eq i32 %n, 0		; <i1>:0 [#uses=2]
	br i1 %cond, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%umax = select i1 %cond, i32 1, i32 %n		; <i32> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	tail call void @bar( i32 %i.03 ) nounwind
	%tmp1 = uitofp i32 %i.03 to double		; <double>:1 [#uses=1]
	tail call void @foo( double %tmp1 ) nounwind
	%indvar.next = add nsw nuw i32 %i.03, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast because the mantissa bits for double are not enough
; to hold all of i64 IV bits.
define void @foobar2(i64 %n) nounwind {

; CHECK-LABEL:  foobar2(
; CHECK-NOT:    phi double
; CHECK-NOT:    phi float

entry:
	%cond = icmp eq i64 %n, 0		; <i1>:0 [#uses=2]
	br i1 %cond, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%umax = select i1 %cond, i64 1, i64 %n		; <i64> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%i.03 = phi i64 [ 0, %bb.nph ], [ %indvar.next, %bb ]		; <i64> [#uses=3]
	%tmp1 = trunc i64 %i.03 to i32		; <i32>:1 [#uses=1]
	tail call void @bar( i32 %tmp1 ) nounwind
	%tmp2 = uitofp i64 %i.03 to double		; <double>:2 [#uses=1]
	tail call void @foo( double %tmp2 ) nounwind
	%indvar.next = add nsw nuw i64 %i.03, 1		; <i64> [#uses=2]
	%exitcond = icmp eq i64 %indvar.next, %umax		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast due to potentional overflow.
define void @foobar3() nounwind {

; CHECK-LABEL:  foobar3(
; CHECK-NOT:    phi double
; CHECK-NOT:    phi float

entry:
	%tmp0 = tail call i32 (...) @nn( ) nounwind		; <i32>:0 [#uses=1]
	%cond = icmp eq i32 %tmp0, 0		; <i1>:1 [#uses=1]
	br i1 %cond, label %return, label %bb

bb:		; preds = %bb, %entry
	%i.03 = phi i32 [ 0, %entry ], [ %indvar.next, %bb ]		; <i32> [#uses=3]
	tail call void @bar( i32 %i.03 ) nounwind
	%tmp2 = uitofp i32 %i.03 to double		; <double>:2 [#uses=1]
	tail call void @foo( double %tmp2 ) nounwind
	%indvar.next = add nuw nsw i32 %i.03, 1		; <i32>:3 [#uses=2]
	%tmp4 = tail call i32 (...) @nn( ) nounwind		; <i32>:4 [#uses=1]
	%exitcond = icmp ugt i32 %tmp4, %indvar.next		; <i1>:5 [#uses=1]
	br i1 %exitcond, label %bb, label %return

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast due to overflow.
define void @foobar4() nounwind {

; CHECK-LABEL:  foobar4(
; CHECK-NOT:    phi double
; CHECK-NOT:    phi float

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
	%indvar.next = add nsw nuw i8 %i.03, 1		; <i32> [#uses=2]
	%tmp = sext i8 %indvar.next to i32
	%exitcond = icmp eq i32 %tmp, 32767		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb

return:		; preds = %bb, %entry
	ret void
}

; Unable to eliminate cast because the integer IV overflows (accum exceeds
; SINT_MAX).

define i32 @foobar5() {
; CHECK-LABEL:  foobar5(
; CHECK-NOT:      phi double
; CHECK-NOT:      phi float
entry:
  br label %loop

loop:
  %accum = phi i32 [ -3220, %entry ], [ %accum.next, %loop ]
  %iv = phi i32 [ 12, %entry ], [ %iv.next, %loop ]
  %tmp1 = sitofp i32 %accum to double
  tail call void @foo( double %tmp1 ) nounwind
  %accum.next = add i32 %accum, 9597741
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp ugt i32 %iv, 235
  br i1 %exitcond, label %exit, label %loop

exit:                                           ; preds = %loop
  ret i32 %accum.next
}

; Can eliminate if we set nsw and, thus, think that we don't overflow SINT_MAX.

define i32 @foobar6() {
; CHECK-LABEL:  foobar6(
; CHECK:          phi double

entry:
  br label %loop

loop:
  %accum = phi i32 [ -3220, %entry ], [ %accum.next, %loop ]
  %iv = phi i32 [ 12, %entry ], [ %iv.next, %loop ]
  %tmp1 = sitofp i32 %accum to double
  tail call void @foo( double %tmp1 ) nounwind
  %accum.next = add nsw i32 %accum, 9597741
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp ugt i32 %iv, 235
  br i1 %exitcond, label %exit, label %loop

exit:                                           ; preds = %loop
  ret i32 %accum.next
}

; Unable to eliminate cast because the integer IV overflows (accum exceeds
; UINT_MAX).

define i32 @foobar7() {
; CHECK-LABEL:  foobar7(
; CHECK-NOT:      phi double
; CHECK-NOT:      phi float
entry:
  br label %loop

loop:
  %accum = phi i32 [ -3220, %entry ], [ %accum.next, %loop ]
  %iv = phi i32 [ 12, %entry ], [ %iv.next, %loop ]
  %tmp1 = uitofp i32 %accum to double
  tail call void @foo( double %tmp1 ) nounwind
  %accum.next = add i32 %accum, 9597741
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp ugt i32 %iv, 235
  br i1 %exitcond, label %exit, label %loop

exit:                                           ; preds = %loop
  ret i32 %accum.next
}

; Can eliminate if we set nuw and, thus, think that we don't overflow UINT_MAX.

define i32 @foobar8() {
; CHECK-LABEL:  foobar8(
; CHECK:          phi double

entry:
  br label %loop

loop:
  %accum = phi i32 [ -3220, %entry ], [ %accum.next, %loop ]
  %iv = phi i32 [ 12, %entry ], [ %iv.next, %loop ]
  %tmp1 = uitofp i32 %accum to double
  tail call void @foo( double %tmp1 ) nounwind
  %accum.next = add nuw i32 %accum, 9597741
  %iv.next = add nuw nsw i32 %iv, 1
  %exitcond = icmp ugt i32 %iv, 235
  br i1 %exitcond, label %exit, label %loop

exit:                                           ; preds = %loop
  ret i32 %accum.next
}

declare void @bar(i32)

declare void @foo(double)

declare i32 @nn(...)
