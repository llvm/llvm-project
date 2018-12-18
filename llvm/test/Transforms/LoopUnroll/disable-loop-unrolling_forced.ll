; RUN: opt -disable-loop-unrolling -O1 -S < %s | FileCheck %s
;
; Check loop unrolling metadata is honored despite automatic unrolling
; being disabled in the pass builder.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @forced(
; CHECK: load
; CHECK: load
define void @forced(i32* nocapture %a) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !{!"llvm.loop.unroll.enable"},
                    !{!"llvm.loop.unroll.count", i32 2}}
