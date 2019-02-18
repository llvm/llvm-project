; RUN: opt -always-inline -globalopt -S < %s | FileCheck %s
;
; static void __attribute__((always_inline)) callee(long n, double A[static const restrict n], long i) {
;   for (long j = 0; j < n; j += 1)
;     A[i * n + j] = 42;
; }
;
; void caller(long n, double A[static const restrict n]) {
;   for (long i = 0; i < n; i += 1)
;     callee(n, A, i);
; }
;
; Check that the access groups (llvm.access.group) are correctly merged.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define internal void @callee(i64 %n, double* noalias nonnull %A, i64 %i) #0 {
entry:
  br label %for.cond

for.cond:
  %j.0 = phi i64 [ 0, %entry ], [ %add1, %for.body ]
  %cmp = icmp slt i64 %j.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %mul = mul nsw i64 %i, %n
  %add = add nsw i64 %mul, %j.0
  %arrayidx = getelementptr inbounds double, double* %A, i64 %add
  store double 4.200000e+01, double* %arrayidx, align 8, !llvm.access.group !6
  %add1 = add nuw nsw i64 %j.0, 1
  br label %for.cond, !llvm.loop !7

for.end:
  ret void
}

attributes #0 = { alwaysinline }

!6 = distinct !{}       ; access group
!7 = distinct !{!7, !9} ; LoopID
!9 = !{!"llvm.loop.parallel_accesses", !6}


define void @caller(i64 %n, double* noalias nonnull %A) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:
  call void @callee(i64 %n, double* %A, i64 %i.0), !llvm.access.group !10
  %add = add nuw nsw i64 %i.0, 1
  br label %for.cond, !llvm.loop !11

for.end:
  ret void
}

!10 = distinct !{}         ; access group
!11 = distinct !{!11, !12} ; LoopID
!12 = !{!"llvm.loop.parallel_accesses", !10}


; CHECK: store double 4.200000e+01, {{.*}} !llvm.access.group ![[ACCESS_GROUP_LIST_3:[0-9]+]]
; CHECK: br label %for.cond.i, !llvm.loop ![[LOOP_INNER:[0-9]+]]
; CHECK: br label %for.cond, !llvm.loop ![[LOOP_OUTER:[0-9]+]]

; CHECK: ![[ACCESS_GROUP_LIST_3]] = !{![[ACCESS_GROUP_INNER:[0-9]+]], ![[ACCESS_GROUP_OUTER:[0-9]+]]}
; CHECK: ![[ACCESS_GROUP_INNER]] = distinct !{}
; CHECK: ![[ACCESS_GROUP_OUTER]] = distinct !{}
; CHECK: ![[LOOP_INNER]] = distinct !{![[LOOP_INNER]], ![[ACCESSES_INNER:[0-9]+]]}
; CHECK: ![[ACCESSES_INNER]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_INNER]]}
; CHECK: ![[LOOP_OUTER]] = distinct !{![[LOOP_OUTER]], ![[ACCESSES_OUTER:[0-9]+]]}
; CHECK: ![[ACCESSES_OUTER]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_OUTER]]}
