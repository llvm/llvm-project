; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -passes=loop-vectorize -tail-predication=enabled -S < %s | \
; RUN:  FileCheck %s -check-prefixes=COMMON,CHECK

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -passes=loop-vectorize -tail-predication=enabled-no-reductions -S < %s | \
; RUN:  FileCheck %s -check-prefixes=COMMON,NORED

; RUN: opt -mtriple=thumbv8.1m.main-arm-eabihf -mattr=+mve.fp -passes=loop-vectorize -tail-predication=force-enabled-no-reductions -S < %s | \
; RUN:  FileCheck %s -check-prefixes=COMMON,NORED

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Check that this reduction is allowed, except when reductions are disable on
; the command line.
;
define dso_local i32 @i32_add_reduction(ptr noalias nocapture readonly %B, i32 %N) local_unnamed_addr #0 {
; COMMON-LABEL: i32_add_reduction(
; COMMON:       entry:
; CHECK:        @llvm.get.active.lane.mask
; NORED-NOT:    @llvm.get.active.lane.mask
; COMMON:       }
entry:
  %cmp6 = icmp sgt i32 %N, 0
  br i1 %cmp6, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  br label %for.body

for.cond.cleanup.loopexit:
  %add.lcssa = phi i32 [ %add, %for.body ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %S.0.lcssa = phi i32 [ 1, %entry ], [ %add.lcssa, %for.cond.cleanup.loopexit ]
  ret i32 %S.0.lcssa

for.body:
  %i.08 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %S.07 = phi i32 [ %add, %for.body ], [ 1, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %i.08
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %S.07
  %inc = add nuw nsw i32 %i.08, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
