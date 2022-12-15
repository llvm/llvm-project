; RUN: opt -mcpu=skx -S -passes=loop-vectorize,instcombine -force-vector-width=8 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Case1: With pragma predicate to force tail-folding.
; All memory opertions are masked.
;void fold_tail(int * restrict p, int * restrict q1, int * restrict q2, int guard) {
;   #pragma clang loop vectorize_predicate(enable)
;   for(int ix=0; ix < 1021; ++ix) {
;     if (ix > guard) {
;       p[ix] = q1[ix] + q2[ix];
;     }
;   }
;}

;CHECK-LABEL: @fold_tail
;CHECK: vector.body:
;CHECK: call <8 x i32> @llvm.masked.load
;CHECK: call <8 x i32> @llvm.masked.load
;CHECK: call void @llvm.masked.store

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @fold_tail(ptr noalias nocapture %p, ptr noalias nocapture readonly %q1, ptr noalias nocapture readonly %q2,
i32 %guard) local_unnamed_addr #0 {
entry:
  %0 = sext i32 %guard to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %cmp1 = icmp sgt i64 %indvars.iv, %0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %arrayidx = getelementptr inbounds i32, ptr %q1, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !2
  %arrayidx3 = getelementptr inbounds i32, ptr %q2, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4, !tbaa !2
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %p, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4, !tbaa !2
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1021
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !8
}

; Case2: With pragma assume_safety both, load and store are masked.
; void assume_safety(int * p, int * q1, int * q2, int guard) {
;   #pragma clang loop vectorize(assume_safety)
;   for(int ix=0; ix < 1021; ++ix) {
;     if (ix > guard) {
;       p[ix] = q1[ix] + q2[ix];
;     }
;   }
;}

;CHECK-LABEL: @assume_safety
;CHECK: vector.body:
;CHECK:  call <8 x i32> @llvm.masked.load
;CHECK:  call void @llvm.masked.store

; Function Attrs: norecurse nounwind uwtable
define void @assume_safety(ptr nocapture, ptr nocapture readonly, ptr nocapture readonly, i32) local_unnamed_addr #0 {
  %5 = sext i32 %3 to i64
  br label %7

; <label>:6:
  ret void

; <label>:7:
  %8 = phi i64 [ 0, %4 ], [ %18, %17 ]
  %9 = icmp sgt i64 %8, %5
  br i1 %9, label %10, label %17

; <label>:10:
  %11 = getelementptr inbounds i32, ptr %1, i64 %8
  %12 = load i32, ptr %11, align 4, !tbaa !2, !llvm.mem.parallel_loop_access !6
  %13 = getelementptr inbounds i32, ptr %2, i64 %8
  %14 = load i32, ptr %13, align 4, !tbaa !2, !llvm.mem.parallel_loop_access !6
  %15 = add nsw i32 %14, %12
  %16 = getelementptr inbounds i32, ptr %0, i64 %8
  store i32 %15, ptr %16, align 4, !tbaa !2, !llvm.mem.parallel_loop_access !6
  br label %17

; <label>:17:
  %18 = add nuw nsw i64 %8, 1
  %19 = icmp eq i64 %18, 1021
  br i1 %19, label %6, label %7, !llvm.loop !6
}

; Case3: With pragma assume_safety and pragma predicate both the store and the
; load are masked.
; void fold_tail_and_assume_safety(int * p, int * q1, int * q2, int guard) {
;   #pragma clang loop vectorize(assume_safety) vectorize_predicate(enable)
;   for(int ix=0; ix < 1021; ++ix) {
;     if (ix > guard) {
;       p[ix] = q1[ix] + q2[ix];
;     }
;   }
;}

;CHECK-LABEL: @fold_tail_and_assume_safety
;CHECK: vector.body:
;CHECK: call <8 x i32> @llvm.masked.load
;CHECK: call <8 x i32> @llvm.masked.load
;CHECK: call void @llvm.masked.store

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @fold_tail_and_assume_safety(ptr noalias nocapture %p, ptr noalias nocapture readonly %q1, ptr noalias nocapture readonly %q2,
i32 %guard) local_unnamed_addr #0 {
entry:
  %0 = sext i32 %guard to i64
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.inc ]
  %cmp1 = icmp sgt i64 %indvars.iv, %0
  br i1 %cmp1, label %if.then, label %for.inc

if.then:
  %arrayidx = getelementptr inbounds i32, ptr %q1, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4, !tbaa !2, !llvm.access.group !10
  %arrayidx3 = getelementptr inbounds i32, ptr %q2, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4, !tbaa !2, !llvm.access.group !10
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %p, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4, !tbaa !2, !llvm.access.group !10
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1021
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !11
}

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.vectorize.enable", i1 true}

!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}

!10 = distinct !{}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.parallel_accesses", !10}
!13 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
