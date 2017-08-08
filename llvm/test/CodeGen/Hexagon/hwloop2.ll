; RUN: llc -disable-lsr -march=hexagon < %s | FileCheck %s

; Test for multiple phis with induction variables.

; CHECK: loop0(.LBB{{.}}_{{.}},r{{[0-9]+}})
; CHECK: endloop0

define i32 @hwloop4(i32* nocapture %s, i32* nocapture %a, i32 %n) {
entry:
  %cmp3 = icmp eq i32 %n, 0
  br i1 %cmp3, label %for.end, label %for.body.lr.ph

for.body.lr.ph:
  %.pre = load i32, i32* %s, align 4
  br label %for.body

for.body:
  %0 = phi i32 [ %.pre, %for.body.lr.ph ], [ %add1, %for.body ]
  %j.05 = phi i32 [ 0, %for.body.lr.ph ], [ %add2, %for.body ]
  %lsr.iv = phi i32 [ %lsr.iv.next, %for.body ], [ %n, %for.body.lr.ph ]
  %lsr.iv1 = phi i32* [ %scevgep, %for.body ], [ %a, %for.body.lr.ph ]
  %1 = load i32, i32* %lsr.iv1, align 4
  %add1 = add nsw i32 %0, %1
  store i32 %add1, i32* %s, align 4
  %add2 = add nsw i32 %j.05, 1
  %lsr.iv.next = add i32 %lsr.iv, -1
  %scevgep = getelementptr i32, i32* %lsr.iv1, i32 1
  %cmp = icmp eq i32 %lsr.iv.next, 0
  br i1 %cmp, label %for.end.loopexit, label %for.body

for.end.loopexit:
  br label %for.end

for.end:
  %j.0.lcssa = phi i32 [ 0, %entry ], [ %add2, %for.end.loopexit ]
  ret i32 %j.0.lcssa
}
