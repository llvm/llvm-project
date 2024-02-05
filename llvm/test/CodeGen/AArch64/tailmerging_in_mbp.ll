; RUN: llc <%s -mtriple=aarch64-eabi -verify-machine-dom-info | FileCheck %s

; CHECK-LABEL: test:
; CHECK-LABEL: %cond.false12.i
; CHECK:         b.gt
; CHECK-NEXT:  LBB0_8:
; CHECK-NEXT:    mov	 x8, x9
; CHECK-NEXT:  LBB0_9:
define i64 @test(i64 %n, ptr %a, ptr %b, ptr %c, ptr %d, ptr %e, ptr %f) {
entry:
  %cmp28 = icmp sgt i64 %n, 1
  br i1 %cmp28, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.lr.ph, %if.end
  %j = phi i64 [ %n, %entry ], [ %div, %if.end ]
  %div = lshr i64 %j, 1
  %a.arrayidx = getelementptr inbounds i64, ptr %a, i64 %div
  %a.j = load i64, ptr %a.arrayidx
  %b.arrayidx = getelementptr inbounds i64, ptr %b, i64 %div
  %b.j = load i64, ptr %b.arrayidx
  %cmp.i = icmp slt i64 %a.j, %b.j
  br i1 %cmp.i, label %for.end.loopexit, label %cond.false.i

cond.false.i:                                     ; preds = %for.body
  %cmp4.i = icmp sgt i64 %a.j, %b.j
  br i1 %cmp4.i, label %if.end, label %cond.false6.i

cond.false6.i:                                    ; preds = %cond.false.i
  %c.arrayidx = getelementptr inbounds i64, ptr %c, i64 %div
  %c.j = load i64, ptr %c.arrayidx
  %d.arrayidx = getelementptr inbounds i64, ptr %d, i64 %div
  %d.j = load i64, ptr %d.arrayidx
  %cmp9.i = icmp slt i64 %c.j, %d.j
  br i1 %cmp9.i, label %for.end.loopexit, label %cond.false11.i

cond.false11.i:                                   ; preds = %cond.false6.i
  %cmp14.i = icmp sgt i64 %c.j, %d.j
  br i1 %cmp14.i, label %if.end, label %cond.false12.i

cond.false12.i:                           ; preds = %cond.false11.i
  %e.arrayidx = getelementptr inbounds i64, ptr %e, i64 %div
  %e.j = load i64, ptr %e.arrayidx
  %f.arrayidx = getelementptr inbounds i64, ptr %f, i64 %div
  %f.j = load i64, ptr %f.arrayidx
  %cmp19.i = icmp sgt i64 %e.j, %f.j
  br i1 %cmp19.i, label %if.end, label %for.end.loopexit

if.end:                                           ; preds = %cond.false12.i, %cond.false11.i, %cond.false.i
  %cmp = icmp ugt i64 %j, 3
  br i1 %cmp, label %for.body, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %cond.false12.i, %cond.false6.i, %for.body, %if.end
  %j.0.lcssa.ph = phi i64 [ %j, %cond.false12.i ], [ %j, %cond.false6.i ], [ %j, %for.body ], [ %div, %if.end ]
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  %j.0.lcssa = phi i64 [ %n, %entry ], [ %j.0.lcssa.ph, %for.end.loopexit ]
  %j.2 = add i64 %j.0.lcssa, %n
  %j.3 = mul i64 %j.2, %n
  %j.4 = add i64 %j.3, 10
  ret i64 %j.4
}
