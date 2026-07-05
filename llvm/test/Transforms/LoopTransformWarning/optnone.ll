; RUN: opt -passes=transform-warning -disable-output -pass-remarks-missed=transform-warning -pass-remarks-analysis=transform-warning < %s 2>&1 | FileCheck -allow-empty %s
;
; Verify that no transformation warnings are emitted for functions with
; 'optnone' attribute.
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @func(ptr nocapture %A, ptr nocapture readonly %B, i32 %Length) #0 {
entry:
  %cmp9 = icmp sgt i32 %Length, 0
  br i1 %cmp9, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %idxprom1 = sext i32 %0 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %idxprom1
  %1 = load i32, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %1, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %Length
  br i1 %exitcond, label %for.end.loopexit, label %for.body, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

attributes #0 = { noinline optnone }

!0 = distinct !{!0, !1, !2, !3}
!1 = !{!"llvm.loop.unroll.enable"}
!2 = !{!"llvm.loop.distribute.enable"}
!3 = !{!"llvm.loop.unroll_and_jam.enable"}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}


; CHECK-NOT: warning
