; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed -polly-simplify -analyze < %s | FileCheck %s
;
; llvm.org/PR33323
;
; Do not remove the pair (store double %add119, read %add119) as redundant
; because the are in the wrong order.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define fastcc void @pr33323([1000 x double]* nocapture %data, [1000 x double]* nocapture %symmat) {
entry:
  br label %for.body98

for.cond87.loopexit:
  ret void

for.body98:
  %indvars.iv13 = phi i64 [ 1, %entry ], [ %indvars.iv.next14, %for.end122 ]
  br label %for.body105

for.body105:
  %indvars.iv = phi i64 [ 0, %for.body98 ], [ %indvars.iv.next, %for.body105 ]
  %arrayidx109 = getelementptr inbounds [1000 x double], [1000 x double]* %data, i64 %indvars.iv, i64 0
  %add119 = fadd double undef, undef
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end122, label %for.body105

for.end122:
  %arrayidx130 = getelementptr inbounds [1000 x double], [1000 x double]* %symmat, i64 %indvars.iv13, i64 0
  store double %add119, double* %arrayidx130
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1
  %exitcond15 = icmp eq i64 %indvars.iv.next14, 1000
  br i1 %exitcond15, label %for.cond87.loopexit, label %for.body98
}


; CHECK: Statistics {
; CHECK:    Redundant writes removed: 1
; CHECK:    Stmts removed: 1
; CHECK: }

; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_for_body105
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_for_body105[i0, i1] -> MemRef_add119[] };
; CHECK-NEXT:            new: { Stmt_for_body105[i0, i1] -> MemRef_symmat[1 + i0, 0] };
; CHECK-NEXT: }
