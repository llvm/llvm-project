; RUN: opt -loops -analyze < %s | FileCheck %s
;
; void func(long n, double A[static const restrict 4*n], double B[static const restrict 4*n]) {
;   for (long i = 0; i < n; i += 1)
;     for (long j = 0; j < n; j += 1)
;       for (long k = 0; k < n; k += 1)
;         for (long l = 0; l < n; l += 1) {
;           A[i + j + k + l] = 21;
;           B[i + j + k + l] = 42;
;         }
; }
;
; Check that isAnnotatedParallel is working as expected.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @func(i64 %n, double* noalias nonnull %A, double* noalias nonnull %B) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i64 [ 0, %entry ], [ %add28, %for.inc27 ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.cond2, label %for.end29

for.cond2:
  %j.0 = phi i64 [ %add25, %for.inc24 ], [ 0, %for.cond ]
  %cmp3 = icmp slt i64 %j.0, %n
  br i1 %cmp3, label %for.cond6, label %for.inc27

for.cond6:
  %k.0 = phi i64 [ %add22, %for.inc21 ], [ 0, %for.cond2 ]
  %cmp7 = icmp slt i64 %k.0, %n
  br i1 %cmp7, label %for.cond10, label %for.inc24

for.cond10:
  %l.0 = phi i64 [ %add20, %for.body13 ], [ 0, %for.cond6 ]
  %cmp11 = icmp slt i64 %l.0, %n
  br i1 %cmp11, label %for.body13, label %for.inc21

for.body13:
  %add = add nuw nsw i64 %i.0, %j.0
  %add14 = add nuw nsw i64 %add, %k.0
  %add15 = add nuw nsw i64 %add14, %l.0
  %arrayidx = getelementptr inbounds double, double* %A, i64 %add15
  store double 2.100000e+01, double* %arrayidx, align 8, !llvm.access.group !5
  %add16 = add nuw nsw i64 %i.0, %j.0
  %add17 = add nuw nsw i64 %add16, %k.0
  %add18 = add nuw nsw i64 %add17, %l.0
  %arrayidx19 = getelementptr inbounds double, double* %B, i64 %add18
  store double 4.200000e+01, double* %arrayidx19, align 8, !llvm.access.group !6
  %add20 = add nuw nsw i64 %l.0, 1
  br label %for.cond10, !llvm.loop !11

for.inc21:
  %add22 = add nuw nsw i64 %k.0, 1
  br label %for.cond6, !llvm.loop !14

for.inc24:
  %add25 = add nuw nsw i64 %j.0, 1
  br label %for.cond2, !llvm.loop !16

for.inc27:
  %add28 = add nuw nsw i64 %i.0, 1
  br label %for.cond, !llvm.loop !18

for.end29:
  ret void
}

; access groups
!7 = distinct !{}
!8 = distinct !{}
!10 = distinct !{}

; access group lists
!5 = !{!7, !10}
!6 = !{!7, !8, !10}

; LoopIDs
!11 = distinct !{!11, !{!"llvm.loop.parallel_accesses", !10}}
!14 = distinct !{!14, !{!"llvm.loop.parallel_accesses", !8, !10}}
!16 = distinct !{!16, !{!"llvm.loop.parallel_accesses", !8}}
!18 = distinct !{!18, !{!"llvm.loop.parallel_accesses", !7}}


; CHECK: Parallel Loop at depth 1
; CHECK-NOT: Parallel
; CHECK:     Loop at depth 2
; CHECK:         Parallel Loop
; CHECK:             Parallel Loop
