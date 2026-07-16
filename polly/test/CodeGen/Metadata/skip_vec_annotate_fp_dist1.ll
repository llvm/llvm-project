; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' -polly-annotate-metadata-vectorize < %s | FileCheck %s

; Verify that vectorize.enable metadata is NOT added for a loop with a dist=1
; dependence involving floating-point operations. This is a workaround for
; https://github.com/llvm/llvm-project/issues/198726 where vectorize.enable
; implicitly allows FP reassociation causing correctness failures.

; void scale(double *A, double factor, int n) {
;   for (int i = 1; i < n; i++)
;     A[i] = factor * A[i - 1];
; }

; The Polly-generated loop should NOT have vectorize.enable metadata.
; CHECK: polly.stmt.for.body:
; CHECK: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
; CHECK: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]],
; CHECK-NOT: !"llvm.loop.vectorize.enable", i1 true

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

define dso_local void @scale(ptr nocapture noundef %A, double noundef %factor, i32 noundef %n) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp5 = icmp sgt i32 %n, 1
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry.split
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry.split
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %0 = add nsw i64 %indvars.iv, -1
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %0
  %1 = load double, ptr %arrayidx, align 8
  %mul = fmul double %factor, %1
  %arrayidx2 = getelementptr inbounds double, ptr %A, i64 %indvars.iv
  store double %mul, ptr %arrayidx2, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !0
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+aes,+crc,+fp-armv8,+neon,+outline-atomics,+perfmon,+sha2,+v8a,-fmv" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
