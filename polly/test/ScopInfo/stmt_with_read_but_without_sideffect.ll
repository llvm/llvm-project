; RUN: opt %loadPolly -polly-delicm -analyze < %s | FileCheck %s
;
; The statement Stmt_for_if_else_1 should be removed because it has no
; sideeffects.  But it has a use of MemRef_tmp21 that must also be
; removed from every list containing it.
; This is a test-case meant for ScopInfo, but only later pass iterate
; over the uses of MemRef_tmp21 of which the use by Stmt_for_if_else_1
; should have been removed. We use -polly-delicm to trigger such an
; iteration of an already deleted MemoryAccess.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@ATH = external dso_local unnamed_addr constant [88 x float], align 16

define void @setup_tone_curves() {
entry:
  %ath = alloca [56 x float], align 16
  br label %for.body

for.cond176.preheader:                            ; preds = %for.cond107.preheader
  unreachable

for.body:                                         ; preds = %for.cond107.preheader, %entry
  %indvars.iv999 = phi i64 [ 0, %entry ], [ %indvars.iv.next1000, %for.cond107.preheader ]
  %tmp5 = shl nsw i64 %indvars.iv999, 2
  br label %for.cond7.preheader

for.cond33.preheader:                             ; preds = %for.inc.1
  br label %for.cond107.preheader

for.cond7.preheader:                              ; preds = %for.inc.1, %for.body
  %indvars.iv958 = phi i64 [ 0, %for.body ], [ %indvars.iv.next959, %for.inc.1 ]
  %tmp20 = add nuw nsw i64 %indvars.iv958, %tmp5
  %arrayidx = getelementptr inbounds [88 x float], [88 x float]* @ATH, i64 0, i64 %tmp20
  %tmp21 = load float, float* %arrayidx, align 4
  %tmp22 = add nuw nsw i64 %tmp20, 1
  %cmp12.1 = icmp ult i64 %tmp22, 88
  br i1 %cmp12.1, label %if.then.1, label %if.else.1

for.cond107.preheader:                            ; preds = %for.cond33.preheader
  %indvars.iv.next1000 = add nuw nsw i64 %indvars.iv999, 1
  br i1 undef, label %for.cond176.preheader, label %for.body

if.else.1:                                        ; preds = %for.cond7.preheader
  %cmp23.1 = fcmp ogt float %tmp21, -3.000000e+01
  br label %for.inc.1

if.then.1:                                        ; preds = %for.cond7.preheader
  %arrayidx.1 = getelementptr inbounds [88 x float], [88 x float]* @ATH, i64 0, i64 %tmp22
  %tmp155 = load float, float* %arrayidx.1, align 4
  %cmp16.1 = fcmp ogt float %tmp21, %tmp155
  br label %for.inc.1

for.inc.1:                                        ; preds = %if.then.1, %if.else.1
  %min.1.1 = phi float [ %tmp155, %if.then.1 ], [ -3.000000e+01, %if.else.1 ]
  %arrayidx.2 = getelementptr inbounds [88 x float], [88 x float]* @ATH, i64 0, i64 0
  %tmp157 = load float, float* %arrayidx.2, align 4
  %cmp16.2 = fcmp ogt float %min.1.1, %tmp157
  %arrayidx.3 = getelementptr inbounds [88 x float], [88 x float]* @ATH, i64 0, i64 0
  %tmp159 = load float, float* %arrayidx.3, align 4
  %cmp16.3 = fcmp ogt float %tmp157, %tmp159
  %arrayidx29 = getelementptr inbounds [56 x float], [56 x float]* %ath, i64 0, i64 %indvars.iv958
  store float %tmp159, float* %arrayidx29, align 4
  %indvars.iv.next959 = add nuw nsw i64 %indvars.iv958, 1
  %exitcond961 = icmp eq i64 %indvars.iv.next959, 56
  br i1 %exitcond961, label %for.cond33.preheader, label %for.cond7.preheader
}


; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_for_cond7_preheader
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_cond7_preheader[i0] -> MemRef_ATH[4p_0 + i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_cond7_preheader[i0] -> MemRef_tmp21[] };
; CHECK-NEXT:            new: [p_0] -> { Stmt_for_cond7_preheader[i0] -> MemRef_ath[i0] };
; CHECK-NEXT:     Stmt_if_then_1
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p_0] -> { Stmt_if_then_1[i0] -> MemRef_ATH[1 + 4p_0 + i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p_0] -> { Stmt_if_then_1[i0] -> MemRef_tmp21[] };
; CHECK-NEXT:            new: [p_0] -> { Stmt_if_then_1[i0] -> MemRef_ath[i0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p_0] -> { Stmt_if_then_1[i0] -> MemRef_min_1_1__phi[] };
; CHECK-NEXT:            new: [p_0] -> { Stmt_if_then_1[i0] -> MemRef_ath[i0] };
; CHECK-NEXT:     Stmt_if_else_1_last
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p_0] -> { Stmt_if_else_1_last[i0] -> MemRef_min_1_1__phi[] };
; CHECK-NEXT:            new: [p_0] -> { Stmt_if_else_1_last[i0] -> MemRef_ath[i0] : p_0 <= 576460752303423487 };
; CHECK-NEXT:     Stmt_for_inc_1
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_inc_1[i0] -> MemRef_min_1_1__phi[] };
; CHECK-NEXT:            new: [p_0] -> { Stmt_for_inc_1[i0] -> MemRef_ath[i0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_inc_1[i0] -> MemRef_ATH[0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_inc_1[i0] -> MemRef_ATH[0] };
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p_0] -> { Stmt_for_inc_1[i0] -> MemRef_ath[i0] };
; CHECK-NEXT: }
