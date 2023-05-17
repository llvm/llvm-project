; RUN: opt -passes=instcombine -S < %s | FileCheck %s
;
; void func(long n, double A[static const restrict n]) {
; 	for (int i = 0; i <  n; i+=1)
; 		for (int j = 0; j <  n;j+=1)
; 			for (int k = 0; k < n; k += 1)
; 				for (int l = 0; l < n; l += 1) {
; 					ptr p = &A[i + j + k + l];
; 					double x = *p;
; 					double y = *p;
; 					arg(x + y);
; 				}
; }
;
; Check for correctly merging access group metadata for instcombine
; (only common loops are parallel == intersection)
; Note that combined load would be parallel to loop !16 since both
; origin loads are parallel to it, but it references two access groups
; (!8 and !9), neither of which contain both loads. As such, the
; information that the combined load is parallel to !16 is lost.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @arg(double)

define void @func(i64 %n, ptr noalias nonnull %A) {
entry:
  br label %for.cond

for.cond:
  %i.0 = phi i32 [ 0, %entry ], [ %add31, %for.inc30 ]
  %conv = sext i32 %i.0 to i64
  %cmp = icmp slt i64 %conv, %n
  br i1 %cmp, label %for.cond2, label %for.end32

for.cond2:
  %j.0 = phi i32 [ %add28, %for.inc27 ], [ 0, %for.cond ]
  %conv3 = sext i32 %j.0 to i64
  %cmp4 = icmp slt i64 %conv3, %n
  br i1 %cmp4, label %for.cond8, label %for.inc30

for.cond8:
  %k.0 = phi i32 [ %add25, %for.inc24 ], [ 0, %for.cond2 ]
  %conv9 = sext i32 %k.0 to i64
  %cmp10 = icmp slt i64 %conv9, %n
  br i1 %cmp10, label %for.cond14, label %for.inc27

for.cond14:
  %l.0 = phi i32 [ %add23, %for.body19 ], [ 0, %for.cond8 ]
  %conv15 = sext i32 %l.0 to i64
  %cmp16 = icmp slt i64 %conv15, %n
  br i1 %cmp16, label %for.body19, label %for.inc24

for.body19:
  %add = add nsw i32 %i.0, %j.0
  %add20 = add nsw i32 %add, %k.0
  %add21 = add nsw i32 %add20, %l.0
  %idxprom = sext i32 %add21 to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %0 = load double, ptr %arrayidx, align 8, !llvm.access.group !1
  %1 = load double, ptr %arrayidx, align 8, !llvm.access.group !2
  %add22 = fadd double %0, %1
  call void @arg(double %add22), !llvm.access.group !3
  %add23 = add nsw i32 %l.0, 1
  br label %for.cond14, !llvm.loop !11

for.inc24:
  %add25 = add nsw i32 %k.0, 1
  br label %for.cond8, !llvm.loop !14

for.inc27:
  %add28 = add nsw i32 %j.0, 1
  br label %for.cond2, !llvm.loop !16

for.inc30:
  %add31 = add nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !18

for.end32:
  ret void
}


; access groups
!7 = distinct !{}
!8 = distinct !{}
!9 = distinct !{}

; access group lists
!1 = !{!7, !9}
!2 = !{!7, !8}
!3 = !{!7, !8, !9}

!11 = distinct !{!11, !13}
!13 = !{!"llvm.loop.parallel_accesses", !7}

!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.parallel_accesses", !8}

!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.parallel_accesses", !8, !9}

!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.parallel_accesses", !9}


; CHECK: load double, {{.*}} !llvm.access.group ![[ACCESSGROUP_0:[0-9]+]]
; CHECK: br label %for.cond14, !llvm.loop ![[LOOP_4:[0-9]+]]

; CHECK: ![[ACCESSGROUP_0]] = distinct !{}

; CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[PARALLEL_ACCESSES_5:[0-9]+]]}
; CHECK: ![[PARALLEL_ACCESSES_5]] = !{!"llvm.loop.parallel_accesses", ![[ACCESSGROUP_0]]}
