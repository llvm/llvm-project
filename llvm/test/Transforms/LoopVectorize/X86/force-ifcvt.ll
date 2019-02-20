; RUN: opt -loop-vectorize -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @Test(i32* nocapture %res, i32* nocapture readnone %c, i32* nocapture readonly %d, i32* nocapture readonly %p) #0 {
entry:
  br label %for.body

; CHECK-LABEL: @Test
; CHECK: <4 x i32>

for.body:                                         ; preds = %cond.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %cond.end ]
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4, !llvm.access.group !1
  %cmp1 = icmp eq i32 %0, 0
  %arrayidx3 = getelementptr inbounds i32, i32* %res, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx3, align 4, !llvm.access.group !1
  br i1 %cmp1, label %cond.end, label %cond.false

cond.false:                                       ; preds = %for.body
  %arrayidx7 = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx7, align 4, !llvm.access.group !1
  %add = add nsw i32 %2, %1
  br label %cond.end

cond.end:                                         ; preds = %for.body, %cond.false
  %cond = phi i32 [ %add, %cond.false ], [ %1, %for.body ]
  store i32 %cond, i32* %arrayidx3, align 4, !llvm.access.group !1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %cond.end
  ret void
}

attributes #0 = { norecurse nounwind uwtable "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }

!0 = distinct !{!0, !{!"llvm.loop.parallel_accesses", !1}}
!1 = distinct !{}
