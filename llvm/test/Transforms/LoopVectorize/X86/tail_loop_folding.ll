; RUN: opt < %s -loop-vectorize -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -prefer-predicate-over-epilog -S | FileCheck -check-prefix=PREDFLAG %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @tail_folding_enabled(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) local_unnamed_addr #0 {
; CHECK-LABEL: tail_folding_enabled(
; CHECK:  vector.body:
; CHECK:  %wide.masked.load = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; CHECK:  %wide.masked.load1 = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; CHECK:  %8 = add nsw <8 x i32> %wide.masked.load1, %wide.masked.load
; CHECK:  call void @llvm.masked.store.v8i32.p0v8i32(
; CHECK:  %index.next = add i64 %index, 8
; CHECK:  %12 = icmp eq i64 %index.next, 432
; CHECK:  br i1 %12, label %middle.block, label %vector.body, !llvm.loop !0
; PREDFLAG-LABEL: tail_folding_enabled(
; PREDFLAG:  vector.body:
; PREDFLAG:  %wide.masked.load = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; PREDFLAG:  %wide.masked.load1 = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; PREDFLAG:  %8 = add nsw <8 x i32> %wide.masked.load1, %wide.masked.load
; PREDFLAG:  call void @llvm.masked.store.v8i32.p0v8i32(
; PREDFLAG:  %index.next = add i64 %index, 8
; PREDFLAG:  %12 = icmp eq i64 %index.next, 432
; PREDFLAG:  br i1 %12, label %middle.block, label %vector.body, !llvm.loop !0
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}

define dso_local void @tail_folding_disabled(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C) local_unnamed_addr #0 {
; CHECK-LABEL: tail_folding_disabled(
; CHECK:      vector.body:
; CHECK-NOT:  @llvm.masked.load.v8i32.p0v8i32(
; CHECK-NOT:  @llvm.masked.store.v8i32.p0v8i32(
; CHECK:      br i1 %44, label {{.*}}, label %vector.body
; PREDFLAG-LABEL: tail_folding_disabled(
; PREDFLAG:  vector.body:
; PREDFLAG:  %wide.masked.load = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; PREDFLAG:  %wide.masked.load1 = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(
; PREDFLAG:  %8 = add nsw <8 x i32> %wide.masked.load1, %wide.masked.load
; PREDFLAG:  call void @llvm.masked.store.v8i32.p0v8i32(
; PREDFLAG:  %index.next = add i64 %index, 8
; PREDFLAG:  %12 = icmp eq i64 %index.next, 432
; PREDFLAG:  br i1 %12, label %middle.block, label %vector.body, !llvm.loop !4
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx2, align 4
  %add = add nsw i32 %1, %0
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %add, i32* %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 430
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !10
}

; Check that fold tail under optsize passes the reduction live-out value
; through a select.
; int reduction_i32(int *A, int *B, int N) {
;   int sum = 0;
;   for (int i = 0; i < N; ++i)
;     sum += (A[i] + B[i]);
;   return sum;
; }
;
define i32 @reduction_i32(i32* nocapture readonly %A, i32* nocapture readonly %B, i32 %N) #0 {
; CHECK-LABEL: @reduction_i32(
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[ACCUM_PHI:%.*]] = phi <8 x i32> [ zeroinitializer, %vector.ph ], [ [[ACCUM:%.*]], %vector.body ]
; CHECK:         [[ICMPULE:%.*]] = icmp ule <8 x i64>
; CHECK:         [[LOAD1:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* {{.*}}, i32 4, <8 x i1> [[ICMPULE]], <8 x i32> undef)
; CHECK:         [[LOAD2:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0v8i32(<8 x i32>* {{.*}}, i32 4, <8 x i1> [[ICMPULE]], <8 x i32> undef)
; CHECK-NEXT:    [[ADD:%.*]] = add nsw <8 x i32> [[LOAD2]], [[LOAD1]]
; CHECK-NEXT:    [[ACCUM]] = add <8 x i32> [[ADD]], [[ACCUM_PHI]]
; CHECK:         [[LIVEOUT:%.*]] = select <8 x i1> [[ICMPULE]], <8 x i32> [[ACCUM]], <8 x i32> [[ACCUM_PHI]]
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 8
; CHECK:       middle.block:
; CHECK-NEXT:    [[RDX_SHUF:%.*]] = shufflevector <8 x i32> [[LIVEOUT]], <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX:%.*]] = add <8 x i32> [[LIVEOUT]], [[RDX_SHUF]]
; CHECK-NEXT:    [[RDX_SHUF4:%.*]] = shufflevector <8 x i32> [[BIN_RDX]], <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX5:%.*]] = add <8 x i32> [[BIN_RDX]], [[RDX_SHUF4]]
; CHECK-NEXT:    [[RDX_SHUF6:%.*]] = shufflevector <8 x i32> [[BIN_RDX5]], <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
; CHECK-NEXT:    [[BIN_RDX7:%.*]] = add <8 x i32> [[BIN_RDX5]], [[RDX_SHUF6]]
; CHECK-NEXT:    [[TMP17:%.*]] = extractelement <8 x i32> [[BIN_RDX7]], i32 0
; CHECK-NEXT:    br i1 true, label %for.cond.cleanup, label %scalar.ph
; CHECK:       scalar.ph:
; CHECK:       for.cond.cleanup:
; CHECK-NEXT:    [[SUM_1_LCSSA:%.*]] = phi i32 [ {{.*}}, %for.body ], [ [[TMP17]], %middle.block ]
; CHECK-NEXT:    ret i32 [[SUM_1_LCSSA]]
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.0 = phi i32 [ %sum.1, %for.body ], [ 0, %entry ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidxA = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidxA, align 4
  %arrayidxB = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %1 = load i32, i32* %arrayidxB, align 4
  %add = add nsw i32 %1, %0
  %sum.1 = add nuw nsw i32 %add, %sum.0
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret i32 %sum.1
}

; CHECK:      !0 = distinct !{!0, !1}
; CHECK-NEXT: !1 = !{!"llvm.loop.isvectorized", i32 1}
; CHECK-NEXT: !2 = distinct !{!2, !3, !1}
; CHECK-NEXT: !3 = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK-NEXT: !4 = distinct !{!4, !1}
; CHECK-NEXT: !5 = distinct !{!5, !3, !1}

attributes #0 = { nounwind optsize uwtable "target-cpu"="core-avx2" "target-features"="+avx,+avx2" }

!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
!8 = !{!"llvm.loop.vectorize.enable", i1 true}

!10 = distinct !{!10, !11, !12}
!11 = !{!"llvm.loop.vectorize.predicate.enable", i1 false}
!12 = !{!"llvm.loop.vectorize.enable", i1 true}
