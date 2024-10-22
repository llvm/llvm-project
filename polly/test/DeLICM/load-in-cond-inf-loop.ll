; RUN: opt %loadNPMPolly '-passes=print<polly-delicm>' -disable-output < %s | FileCheck %s

; When %b is 0, %for.body13 is an infite loop. In this case the loaded
; value %1 is not used anywhere.
; This is a problem when DeLICM tries to map %1 to %arrayidx16 because
; %1 has no corresponding when %b == 0 and therefore hat no location
; where it can be mapped to. However, since %b == 0 results in an
; infinite loop, it should not in the Context, or in this case, in the
; InvalidContext.
;
; Test case reduced from llvm.org/PR48445.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@arr_18 = external dso_local local_unnamed_addr global [0 x i16], align 2

define void @func(i64 %b, ptr %c) {
entry:
  %conv1 = trunc i64 %b to i32
  %sext = shl i32 %conv1, 24
  %conv2 = ashr exact i32 %sext, 24
  %arrayidx = getelementptr inbounds i8, ptr %c, i64 %b
  %tobool19.not = icmp eq i64 %b, 0
  br label %for.cond3.preheader

for.cond3.preheader:
  %d.039 = phi i16 [ 0, %entry ], [ %inc, %for.cond.cleanup6 ]
  %idxprom = sext i16 %d.039 to i64
  br label %for.cond8.preheader

for.cond8.preheader:
  br label %for.body13

for.cond.cleanup6:
  %arrayidx16 = getelementptr inbounds [0 x i16], ptr @arr_18, i64 0, i64 %idxprom
  %0 = zext i8 %1 to i16
  store i16 %0, ptr %arrayidx16, align 2
  %inc = add i16 %d.039, 1
  %conv = sext i16 %inc to i32
  %cmp = icmp sgt i32 %conv2, %conv
  br i1 %cmp, label %for.cond3.preheader, label %for.cond.cleanup

for.cond.cleanup12:
  br i1 false, label %for.cond8.preheader, label %for.cond.cleanup6

for.body13:
  %1 = load i8, ptr %arrayidx, align 1
  br i1 %tobool19.not, label %for.body13, label %for.cond.cleanup12

for.cond.cleanup:
  ret void
}


; CHECK: Statistics {
; CHECK:     Value scalars mapped:  1
; CHECK: }
; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_for_body13
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [b] -> { Stmt_for_body13[i0, i1, i2] -> MemRef_c[b] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [b] -> { Stmt_for_body13[i0, i1, i2] -> MemRef1[] };
; CHECK-NEXT:            new: [b] -> { Stmt_for_body13[i0, i1, i2] -> MemRef_arr_18[i0] : i0 < b; Stmt_for_body13[0, i1, i2] -> MemRef_arr_18[0] : b < 0 };
; CHECK-NEXT:     Stmt_for_cond_cleanup6
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [b] -> { Stmt_for_cond_cleanup6[i0] -> MemRef1[] };
; CHECK-NEXT:            new: [b] -> { Stmt_for_cond_cleanup6[i0] -> MemRef_arr_18[i0] : i0 < b; Stmt_for_cond_cleanup6[0] -> MemRef_arr_18[0] : b < 0 };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [b] -> { Stmt_for_cond_cleanup6[i0] -> MemRef_arr_18[i0] };
; CHECK-NEXT: }
