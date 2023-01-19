; RUN: opt -opaque-pointers=0 %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -S < %s | FileCheck %s
;
; A load must provide a value for every statement instance.
; Statement instances not in the SCoP's context are irrelevant.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@ATH = external dso_local unnamed_addr constant [88 x float], align 16

define void @load_subset_with_context() {
entry:
  %ath = alloca [56 x float], align 16
  br label %for.body

for.cond176.preheader:                            ; preds = %for.cond33.preheader
  ret void

for.body:                                         ; preds = %for.cond33.preheader, %entry
  %indvars.iv999 = phi i64 [ 0, %entry ], [ %indvars.iv.next1000, %for.cond33.preheader ]
  %tmp5 = shl nsw i64 %indvars.iv999, 2
  br label %for.cond7.preheader

for.cond33.preheader:                             ; preds = %for.inc.3
  %tmp175 = load float, float* undef, align 4
  %indvars.iv.next1000 = add nuw nsw i64 %indvars.iv999, 1
  %exitcond1002 = icmp eq i64 %indvars.iv.next1000, 17
  br i1 %exitcond1002, label %for.cond176.preheader, label %for.body

for.cond7.preheader:                              ; preds = %for.inc.3, %for.body
  %indvars.iv958 = phi i64 [ 0, %for.body ], [ %indvars.iv.next959, %for.inc.3 ]
  %tmp20 = add nuw nsw i64 %indvars.iv958, %tmp5
  %arrayidx.2 = getelementptr inbounds [88 x float], [88 x float]* @ATH, i64 0, i64 0
  %tmp157 = load float, float* %arrayidx.2, align 4
  %tmp158 = add nuw nsw i64 %tmp20, 3
  %cmp12.3 = icmp ult i64 %tmp158, 88
  br i1 %cmp12.3, label %if.then.3, label %if.else.3

if.else.3:                                        ; preds = %for.cond7.preheader
  br label %for.inc.3

if.then.3:                                        ; preds = %for.cond7.preheader
  br label %for.inc.3

for.inc.3:                                        ; preds = %if.then.3, %if.else.3
  %min.1.3 = phi float [ undef, %if.then.3 ], [ %tmp157, %if.else.3 ]
  %arrayidx29 = getelementptr inbounds [56 x float], [56 x float]* %ath, i64 0, i64 %indvars.iv958
  store float %min.1.3, float* %arrayidx29, align 4
  %indvars.iv.next959 = add nuw nsw i64 %indvars.iv958, 1
  %exitcond961 = icmp eq i64 %indvars.iv.next959, 56
  br i1 %exitcond961, label %for.cond33.preheader, label %for.cond7.preheader
}


; CHECK:      polly.stmt.if.else.3:
; CHECK-NEXT:   %polly.access.cast.ath1 = bitcast [56 x float]* %ath to float*
; CHECK-NEXT:   %polly.access.ath2 = getelementptr float, float* %polly.access.cast.ath1, i64 %polly.indvar
; CHECK-NEXT:   %polly.access.ath2.reload = load float, float* %polly.access.ath2
