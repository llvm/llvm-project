; Test that MergeConsecutiveStores() does not during DAG combining
; incorrectly drop a chain dependency to a store previously chained to
; one of two combined loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

@A = common global [2048 x float] zeroinitializer, align 4

; Function Attrs: nounwind
define signext i32 @main(i32 signext %argc, i8** nocapture readnone %argv) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv24 = phi i64 [ 0, %entry ], [ %indvars.iv.next25, %for.body ]
  %sum.018 = phi float [ 0.000000e+00, %entry ], [ %add, %for.body ]
  %0 = trunc i64 %indvars.iv24 to i32
  %conv = sitofp i32 %0 to float
  %arrayidx = getelementptr inbounds [2048 x float], [2048 x float]* @A, i64 0, i64 %indvars.iv24
  store float %conv, float* %arrayidx, align 4
  %add = fadd float %sum.018, %conv
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %exitcond26 = icmp eq i64 %indvars.iv.next25, 2048
  br i1 %exitcond26, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  br label %for.body.3.lr.ph.i.preheader

for.body.3.lr.ph.i.preheader:                     ; preds = %complex_transpose.exit, %for.end
  %i.116 = phi i32 [ 0, %for.end ], [ %inc9, %complex_transpose.exit ]
  br label %for.body.3.lr.ph.i

for.body.3.lr.ph.i:                               ; preds = %for.body.3.lr.ph.i.preheader, %for.inc.40.i
  %indvars.iv19 = phi i32 [ 1, %for.body.3.lr.ph.i.preheader ], [ %indvars.iv.next20, %for.inc.40.i ]
  %indvars.iv57.i = phi i64 [ 1, %for.body.3.lr.ph.i.preheader ], [ %indvars.iv.next58.i, %for.inc.40.i ]
  %1 = shl nsw i64 %indvars.iv57.i, 1
  %2 = shl nsw i64 %indvars.iv57.i, 6
  br label %for.body.3.i

for.body.3.i:                                     ; preds = %for.body.3.i, %for.body.3.lr.ph.i
; CHECK-LABEL: .LBB0_5:
; CHECK-NOT:    stfh    %r{{.*}}, 0(%r{{.*}})
; CHECK:        lg      %r{{.*}}, -4(%r{{.*}})
; Overlapping load should go before the store
  %indvars.iv.i = phi i64 [ 0, %for.body.3.lr.ph.i ], [ %indvars.iv.next.i, %for.body.3.i ]
  %3 = shl nsw i64 %indvars.iv.i, 6
  %4 = add nuw nsw i64 %3, %1
  %arrayidx.i = getelementptr inbounds [2048 x float], [2048 x float]* @A, i64 0, i64 %4
  %5 = bitcast float* %arrayidx.i to i32*
  %6 = load i32, i32* %5, align 4
  %arrayidx9.i = getelementptr inbounds float, float* getelementptr inbounds ([2048 x float], [2048 x float]* @A, i64 0, i64 1), i64 %4
  %7 = bitcast float* %arrayidx9.i to i32*
  %8 = load i32, i32* %7, align 4
  %9 = shl nsw i64 %indvars.iv.i, 1
  %10 = add nuw nsw i64 %9, %2
  %arrayidx14.i = getelementptr inbounds [2048 x float], [2048 x float]* @A, i64 0, i64 %10
  %11 = bitcast float* %arrayidx14.i to i32*
  %12 = load i32, i32* %11, align 4
  %arrayidx19.i = getelementptr inbounds float, float* getelementptr inbounds ([2048 x float], [2048 x float]* @A, i64 0, i64 1), i64 %10
  %13 = bitcast float* %arrayidx19.i to i32*
  %14 = load i32, i32* %13, align 4
  store i32 %6, i32* %11, align 4
  store i32 %8, i32* %13, align 4
  store i32 %12, i32* %5, align 4
  store i32 %14, i32* %7, align 4
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next.i to i32
  %exitcond21 = icmp eq i32 %lftr.wideiv, %indvars.iv19
  br i1 %exitcond21, label %for.inc.40.i, label %for.body.3.i

for.inc.40.i:                                     ; preds = %for.body.3.i
  %indvars.iv.next58.i = add nuw nsw i64 %indvars.iv57.i, 1
  %indvars.iv.next20 = add nuw nsw i32 %indvars.iv19, 1
  %exitcond22 = icmp eq i64 %indvars.iv.next58.i, 32
  br i1 %exitcond22, label %complex_transpose.exit, label %for.body.3.lr.ph.i

complex_transpose.exit:                           ; preds = %for.inc.40.i
  %inc9 = add nuw nsw i32 %i.116, 1
  %exitcond23 = icmp eq i32 %inc9, 10
  br i1 %exitcond23, label %for.body.14.preheader, label %for.body.3.lr.ph.i.preheader

for.body.14.preheader:                            ; preds = %complex_transpose.exit
  br label %for.body.14

for.body.14:                                      ; preds = %for.body.14.preheader, %for.body.14
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body.14 ], [ 0, %for.body.14.preheader ]
  %sum.115 = phi float [ %add17, %for.body.14 ], [ 0.000000e+00, %for.body.14.preheader ]
  %arrayidx16 = getelementptr inbounds [2048 x float], [2048 x float]* @A, i64 0, i64 %indvars.iv
  %15 = load float, float* %arrayidx16, align 4
  %add17 = fadd float %sum.115, %15
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 2048
  br i1 %exitcond, label %for.end.20, label %for.body.14

for.end.20:                                       ; preds = %for.body.14
  ret i32 0
}
