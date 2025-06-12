; RUN: opt %loadNPMPolly -S -passes=polly-codegen -polly-annotate-metadata-vectorize < %s | FileCheck %s

; Basic verification of vectorize metadata getting added when "-polly-vectorize-metadata" is
; passed.

; void add(int *A, int *B, int *C,int n) {
;    for(int i=0; i<n; i++)
;      C[i] += A[i] + B[i];
; }

; CHECK: for.body:
; CHECK: br {{.*}} !llvm.loop [[LOOP:![0-9]+]]
; CHECK: polly.stmt.for.body:
; CHECK: br {{.*}} !llvm.loop [[POLLY_LOOP:![0-9]+]]
; CHECK: [[LOOP]] = distinct !{[[LOOP]], [[META2:![0-9]+]], [[META3:![0-9]+]]}
; CHECK: [[META3]] = !{!"llvm.loop.vectorize.enable", i32 0}
; CHECK: [[POLLY_LOOP]] = distinct !{[[POLLY_LOOP]], [[META2:![0-9]+]], [[META3:![0-9]+]]}
; CHECK: [[META3]] = !{!"llvm.loop.vectorize.enable", i1 true}

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @add(ptr nocapture noundef readonly %A, ptr nocapture noundef readonly %B, ptr nocapture noundef %C, i32 noundef %n) local_unnamed_addr #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp10 = icmp sgt i32 %n, 0
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry.split
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry.split
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx4, align 4
  %add5 = add nsw i32 %add, %2
  store i32 %add5, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body, !llvm.loop !0
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-a57" "target-features"="+aes,+crc,+fp-armv8,+neon,+outline-atomics,+perfmon,+sha2,+v8a,-fmv" }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
