; RUN: opt -loop-unroll -unroll-count=2 -S < %s | FileCheck %s
;
; Check whether the llvm.loop.unroll.count loop property overrides
; llvm.loop.disable_nonforced.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @disable_nonforced_count(
; CHECK: store
; CHECK: store
; CHECK-NOT: store
define void @disable_nonforced_count(i32* nocapture %a) {
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

!0 = !{!0, !{!"llvm.loop.disable_nonforced"}, !{!"llvm.loop.unroll.count", i32 2}}
