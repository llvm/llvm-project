; RUN: llc -mtriple=thumbv7-apple-ios -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-T
; RUN: llc -mtriple=armv7-apple-ios   -disable-block-placement < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-A

; LSR should compare against the post-incremented induction variable.
; In this case, the immediate value is -2 which requires a cmn instruction.
;
; CHECK-LABEL: f:
; CHECK: %for.body
; CHECK: sub{{.*}}[[IV:r[0-9]+]], #2
; CHECK-T: adds{{.*}}[[IV]], #2
; CHECK-A: cmn{{.*}}[[IV]], #2
; CHECK: bne
define i32 @f(i32* nocapture %a, i32 %i) nounwind readonly ssp {
entry:
  %cmp3 = icmp eq i32 %i, -2
  br i1 %cmp3, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %bi.06 = phi i32 [ %i.addr.0.bi.0, %for.body ], [ 0, %entry ]
  %i.addr.05 = phi i32 [ %sub, %for.body ], [ %i, %entry ]
  %b.04 = phi i32 [ %.b.0, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %i.addr.05
  %0 = load i32, i32* %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, %b.04
  %.b.0 = select i1 %cmp1, i32 %0, i32 %b.04
  %i.addr.0.bi.0 = select i1 %cmp1, i32 %i.addr.05, i32 %bi.06
  %sub = add nsw i32 %i.addr.05, -2
  %cmp = icmp eq i32 %i.addr.05, 0
  br i1 %cmp, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %bi.0.lcssa = phi i32 [ 0, %entry ], [ %i.addr.0.bi.0, %for.body ]
  ret i32 %bi.0.lcssa
}
