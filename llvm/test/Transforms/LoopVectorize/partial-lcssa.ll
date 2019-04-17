; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s
; We vectorize the inner loop, so we have to put it in LCSSA form.
; However, there's no reason to touch the outer loop.

; CHECK-LABEL: @foo
; CHECK-LABEL: for.end.inner.loopexit:
; CHECK: %[[LCSSAPHI:.*]] = phi i64 [ %indvars.iv, %for.body.inner ], [ %{{.*}}, %middle.block ]
; CHECK: store i64 %[[LCSSAPHI]], i64* %O1, align 4
; CHECK-LABEL: for.end.outer.loopexit
; CHECK: store i64 %indvars.outer, i64* %O2, align 4


define i64 @foo(i32* nocapture %A, i32* nocapture %B, i64 %n, i64 %m, i64* %O1, i64* %O2) {
entry:
  %cmp = icmp sgt i64 %n, 0
  br i1 %cmp, label %for.body.outer.preheader, label %for.end.outer

for.body.outer.preheader:                         ; preds = %entry
  br label %for.body.outer

for.body.outer:                                   ; preds = %for.body.outer.preheader, %for.end.inner
  %indvars.outer = phi i64 [ %indvars.outer.next, %for.end.inner ], [ 0, %for.body.outer.preheader ]
  %cmp2 = icmp sgt i64 %m, 0
  br i1 %cmp2, label %for.body.inner.preheader, label %for.end.inner

for.body.inner.preheader:                         ; preds = %for.body.outer
  br label %for.body.inner

for.body.inner:                                   ; preds = %for.body.inner.preheader, %for.body.inner
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body.inner ], [ 0, %for.body.inner.preheader ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %v = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  store i32 %v, i32* %arrayidx2, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, %n
  br i1 %exitcond, label %for.end.inner.loopexit, label %for.body.inner

for.end.inner.loopexit:                           ; preds = %for.body.inner
  store i64 %indvars.iv, i64 *%O1, align 4
  br label %for.end.inner

for.end.inner:                                    ; preds = %for.end.inner.loopexit, %for.body.outer
  %indvars.outer.next = add i64 %indvars.outer, 1
  %exitcond.outer = icmp eq i64 %indvars.outer, %m
  br i1 %exitcond.outer, label %for.end.outer.loopexit, label %for.body.outer

for.end.outer.loopexit:                           ; preds = %for.end.inner
  store i64 %indvars.outer, i64 *%O2, align 4
  br label %for.end.outer

for.end.outer:                                    ; preds = %for.end.outer.loopexit, %entry
  ret i64 undef
}
