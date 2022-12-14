; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S < %s | FileCheck %s
;
; Check whether the llvm.loop.vectorize.enable loop property overrides
; llvm.loop.disable_nonforced.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; CHECK-LABEL: @disable_nonforced_enable(
; CHECK: store <2 x i32>
define void @disable_nonforced_enable(ptr nocapture %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, ptr %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = !{!0, !{!"llvm.loop.disable_nonforced"}, !{!"llvm.loop.vectorize.enable", i32 1}}
