; RUN: opt -S -passes=loop-vectorize,instsimplify -force-vector-interleave=1 \
; RUN:   -mcpu=neoverse-v1 -sve-tail-folding=disabled < %s | FileCheck %s --check-prefix=CHECK-EPILOG
; RUN: opt -S -passes=loop-vectorize,instsimplify -force-vector-interleave=1 \
; RUN:   -mcpu=neoverse-v2 < %s | FileCheck %s --check-prefix=CHECK-EPILOG-V2
; RUN: opt -S -passes=loop-vectorize,instsimplify -force-vector-interleave=1 \
; RUN:   -mcpu=cortex-x2 < %s | FileCheck %s --check-prefix=CHECK-NO-EPILOG

target triple = "aarch64-unknown-linux-gnu"

define void @foo(ptr noalias nocapture readonly %p, ptr noalias nocapture %q, i64 %len) #0 {
; CHECK-EPILOG:      vec.epilog.ph:
; CHECK-EPILOG:      vec.epilog.vector.body:
; CHECK-EPILOG:        load <vscale x 4 x i16>

; The epilogue loop gets vectorised vscale x 2 x i16 wide.
; CHECK-EPILOG-V2:      vec.epilog.ph:
; CHECK-EPILOG-V2:      vec.epilog.vector.body:
; CHECK-EPILOG-V2:        load <vscale x 2 x i16>

; CHECK-NO-EPILOG-NOT:  vec.epilog.vector.ph:
; CHECK-NO-EPILOG-NOT:  vec.epilog.vector.body:
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i16, ptr %p, i64 %indvars.iv
  %0 = load i16, ptr %arrayidx
  %add = add nuw nsw i16 %0, 2
  %arrayidx3 = getelementptr inbounds i16, ptr %q, i64 %indvars.iv
  store i16 %add, ptr %arrayidx3
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %len
  br i1 %exitcond, label %exit, label %for.body

exit:                                 ; preds = %for.body
  ret void
}

attributes #0 = { "target-features"="+sve" vscale_range(1,16) }
