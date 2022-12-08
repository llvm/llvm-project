; RUN: opt -S < %s -passes='loop-mssa(loop-rotate,licm)'  -verify-dom-info -verify-loop-info | FileCheck %s
; PR2624 unroll multiple exits

@mode_table = global [4 x i32] zeroinitializer		; <[4 x i32]*> [#uses=1]

; CHECK-LABEL: @f(
; CHECK-NOT: bb:
define i8 @f() {
entry:
	tail call i32 @fegetround( )		; <i32>:0 [#uses=1]
	br label %bb

bb:		; preds = %bb4, %entry
	%mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb4 ]		; <i8> [#uses=4]
	zext i8 %mode.0 to i32		; <i32>:1 [#uses=1]
	getelementptr [4 x i32], [4 x i32]* @mode_table, i32 0, i32 %1		; <i32*>:2 [#uses=1]
	load i32, i32* %2, align 4		; <i32>:3 [#uses=1]
	icmp eq i32 %3, %0		; <i1>:4 [#uses=1]
	br i1 %4, label %bb1, label %bb2

bb1:		; preds = %bb
	ret i8 %mode.0

bb2:		; preds = %bb
	icmp eq i8 %mode.0, 1		; <i1>:5 [#uses=1]
	br i1 %5, label %bb5, label %bb4

bb4:		; preds = %bb2
	%indvar.next = add i8 %mode.0, 1		; <i8> [#uses=1]
	br label %bb

bb5:		; preds = %bb2
	tail call void @raise_exception( ) noreturn
	unreachable
}

declare i32 @fegetround()

declare void @raise_exception() noreturn

;CHECK: for.body.lr.ph:
;CHECK-NEXT:  %arrayidx1 = getelementptr inbounds i8, i8* %CurPtr, i64 0
;CHECK-NEXT:  %0 = load i8, i8* %arrayidx1, align 1
;CHECK-NEXT:  %conv2 = sext i8 %0 to i32
;CHECK-NEXT:  br label %for.body

define i32 @foo(i8* %CurPtr, i32 %a) #0 {
entry:
  br label %for.cond

for.cond:					  ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 1, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ne i32 %i.0, %a
  br i1 %cmp, label %for.body, label %return

for.body:					  ; preds = %for.cond
  %idxprom = zext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds i8, i8* %CurPtr, i64 %idxprom
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  %arrayidx1 = getelementptr inbounds i8, i8* %CurPtr, i64 0
  %1 = load i8, i8* %arrayidx1, align 1
  %conv2 = sext i8 %1 to i32
  %cmp3 = icmp ne i32 %conv, %conv2
  br i1 %cmp3, label %return, label %for.inc

for.inc:					  ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

return:						  ; preds = %for.cond, %for.body
  %retval.0 = phi i32 [ 0, %for.body ], [ 1, %for.cond ]
  ret i32 %retval.0
}

attributes #0 = { nounwind uwtable }
