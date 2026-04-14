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

define void @fold_tail(ptr noalias nocapture %p, ptr noalias nocapture readonly %q1, ptr noalias nocapture readonly %q2,
i32 %guard) #0 {
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
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %q2, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %p, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4
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

define void @assume_safety(ptr nocapture, ptr nocapture readonly, ptr nocapture readonly, i32) #0 {
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
  %12 = load i32, ptr %11, align 4, !llvm.mem.parallel_loop_access !6
  %13 = getelementptr inbounds i32, ptr %2, i64 %8
  %14 = load i32, ptr %13, align 4, !llvm.mem.parallel_loop_access !6
  %15 = add nsw i32 %14, %12
  %16 = getelementptr inbounds i32, ptr %0, i64 %8
  store i32 %15, ptr %16, align 4, !llvm.mem.parallel_loop_access !6
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

define void @fold_tail_and_assume_safety(ptr noalias nocapture %p, ptr noalias nocapture readonly %q1, ptr noalias nocapture readonly %q2,
i32 %guard) #0 {
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
  %1 = load i32, ptr %arrayidx, align 4, !llvm.access.group !10
  %arrayidx3 = getelementptr inbounds i32, ptr %q2, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx3, align 4, !llvm.access.group !10
  %add = add nsw i32 %2, %1
  %arrayidx5 = getelementptr inbounds i32, ptr %p, i64 %indvars.iv
  store i32 %add, ptr %arrayidx5, align 4, !llvm.access.group !10
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1021
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !11
}

attributes #0 = { "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "use-soft-float"="false" }

!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.vectorize.enable", i1 true}

!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}

!10 = distinct !{}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.parallel_accesses", !10}
!13 = !{!"llvm.loop.vectorize.predicate.enable", i1 true}
