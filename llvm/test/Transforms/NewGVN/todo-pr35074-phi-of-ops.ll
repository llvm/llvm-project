; REQUIRES: asserts

; RUN: opt -passes=newgvn -S %s | FileCheck %s

XFAIL: *

; TODO: Test case for PR35074. Crashes caused by phi-of-ops.
define void @crash1_pr35074(i32 %this, i1 %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %y.0 = phi i32 [ 1, %entry ], [ %inc7, %for.inc6 ]
  br i1 %c, label %for.inc6, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond
  %sub = add nsw i32 %y.0, -1
  br label %for.body4

for.body4:                                        ; preds = %for.body.lr.ph
  %cmp = icmp ugt i32 %sub, %y.0
  br i1 %cmp, label %for.end, label %for.body4.1

for.end:                                          ; preds = %for.body4.1, %for.body4
  ret void

for.inc6:                                         ; preds = %for.cond
  %inc7 = add nuw nsw i32 %y.0, 1
  br label %for.cond

for.body4.1:                                      ; preds = %for.body4
  %inc.1 = add nuw nsw i32 %y.0, 1
  tail call void @_blah(i32 %inc.1)
  br label %for.end
}

declare void @_blah(i32)
