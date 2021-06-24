; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; ScalarEvolution should be able to understand the loop and eliminate the casts.

target datalayout = "e-p:32:32:32-p1:16:16:16-p2:8:8:8-p4:64:64:64-n16:32:64"

; CHECK:  {%d,+,4}<%bb>{{ U: [^ ]+ S: [^ ]+}}{{ *}} Exits: (-4 + (4 * (trunc i32 %n to i16)) + %d)


define void @foo(i32 addrspace(1)* nocapture %d, i32 %n) nounwind {
; CHECK: @foo
entry:
	%0 = icmp sgt i32 %n, 0		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %return

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%i.02 = phi i32 [ %5, %bb1 ], [ 0, %bb.nph ]		; <i32> [#uses=2]
	%p.01 = phi i8 [ %4, %bb1 ], [ -1, %bb.nph ]		; <i8> [#uses=2]
	%1 = sext i8 %p.01 to i32		; <i32> [#uses=1]
	%2 = sext i32 %i.02 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32, i32 addrspace(1)* %d, i64 %2		; <i32*> [#uses=1]
	store i32 %1, i32 addrspace(1)* %3, align 4
	%4 = add i8 %p.01, 1		; <i8> [#uses=1]
	%5 = add i32 %i.02, 1		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%6 = icmp slt i32 %5, %n		; <i1> [#uses=1]
	br i1 %6, label %bb, label %bb1.return_crit_edge

bb1.return_crit_edge:		; preds = %bb1
	br label %return

return:		; preds = %bb1.return_crit_edge, %entry
	ret void
}

define void @test(i8 addrspace(1)* %a, i32 %n) nounwind {
; CHECK: @test
entry:
  %cmp1 = icmp sgt i32 %n, 0
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %tmp = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ %indvar.next, %for.body ], [ 0, %for.body.lr.ph ]
  %arrayidx = getelementptr i8, i8 addrspace(1)* %a, i64 %indvar
  store i8 0, i8 addrspace(1)* %arrayidx, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, %tmp
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

; CHECK: Determining loop execution counts for: @test
; CHECK-NEXT: backedge-taken count is
; CHECK-NEXT: max backedge-taken count is 2147483646
