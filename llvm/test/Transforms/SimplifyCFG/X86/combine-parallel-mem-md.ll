; RUN: opt -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind uwtable
define void @Test(i32* nocapture %res, i32* nocapture readnone %c, i32* nocapture readonly %d, i32* nocapture readonly %p) #0 {
entry:
  br label %for.body

; CHECK-LABEL: @Test
; CHECK: load i32, i32* {{.*}}, align 4, !llvm.access.group !0
; CHECK: load i32, i32* {{.*}}, align 4, !llvm.access.group !0
; CHECK: store i32 {{.*}}, align 4, !llvm.access.group !0
; CHECK-NOT: load
; CHECK-NOT: store

for.body:                                         ; preds = %cond.end, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %cond.end ]
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4, !llvm.access.group !0
  %cmp1 = icmp eq i32 %0, 0
  br i1 %cmp1, label %cond.true, label %cond.false

cond.false:                                       ; preds = %for.body
  %arrayidx3 = getelementptr inbounds i32, i32* %res, i64 %indvars.iv
  %v = load i32, i32* %arrayidx3, align 4, !llvm.access.group !0
  %arrayidx7 = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx7, align 4, !llvm.access.group !0
  %add = add nsw i32 %1, %v
  br label %cond.end

cond.true:                                       ; preds = %for.body
  %arrayidx4 = getelementptr inbounds i32, i32* %res, i64 %indvars.iv
  %w = load i32, i32* %arrayidx4, align 4, !llvm.access.group !0
  %arrayidx8 = getelementptr inbounds i32, i32* %d, i64 %indvars.iv
  %2 = load i32, i32* %arrayidx8, align 4, !llvm.access.group !0
  %add2 = add nsw i32 %2, %w
  br label %cond.end

cond.end:                                         ; preds = %for.body, %cond.false
  %cond = phi i32 [ %add, %cond.false ], [ %add2, %cond.true ]
  %arrayidx9 = getelementptr inbounds i32, i32* %res, i64 %indvars.iv
  store i32 %cond, i32* %arrayidx9, align 4, !llvm.access.group !0
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %cond.end
  ret void
}

attributes #0 = { norecurse nounwind uwtable }

!0 = distinct !{!0, !1, !{!"llvm.loop.parallel_accesses", !10}}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
!10 = distinct !{}
