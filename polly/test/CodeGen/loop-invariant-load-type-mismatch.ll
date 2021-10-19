; RUN: opt %loadPolly -polly-codegen < %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Just make sure this test passes correctly.

define void @kernel_ludcmp(double* %b, double* %y) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.cond.30.for.cond.loopexit_crit_edge:          ; preds = %for.end.54
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.cond.30.preheader, %for.cond.30.for.cond.loopexit_crit_edge
  %indvars.iv.next131 = add nuw nsw i32 %indvars.iv130, 1
  br i1 false, label %for.body, label %for.end.65

for.body:                                         ; preds = %for.cond.loopexit, %entry.split
  %indvars.iv130 = phi i32 [ 1, %entry.split ], [ %indvars.iv.next131, %for.cond.loopexit ]
  br i1 true, label %for.body.3.lr.ph, label %for.cond.30.preheader

for.body.3.lr.ph:                                 ; preds = %for.body
  br label %for.body.3

for.cond.1.for.cond.30.preheader_crit_edge:       ; preds = %for.end
  br label %for.cond.30.preheader

for.cond.30.preheader:                            ; preds = %for.cond.1.for.cond.30.preheader_crit_edge, %for.body
  br i1 true, label %for.body.32.lr.ph, label %for.cond.loopexit

for.body.32.lr.ph:                                ; preds = %for.cond.30.preheader
  br label %for.body.32

for.body.3:                                       ; preds = %for.end, %for.body.3.lr.ph
  br i1 false, label %for.body.9.lr.ph, label %for.end

for.body.9.lr.ph:                                 ; preds = %for.body.3
  br label %for.body.9

for.body.9:                                       ; preds = %for.body.9, %for.body.9.lr.ph
  br i1 false, label %for.body.9, label %for.cond.7.for.end_crit_edge

for.cond.7.for.end_crit_edge:                     ; preds = %for.body.9
  br label %for.end

for.end:                                          ; preds = %for.cond.7.for.end_crit_edge, %for.body.3
  br i1 false, label %for.body.3, label %for.cond.1.for.cond.30.preheader_crit_edge

for.body.32:                                      ; preds = %for.end.54, %for.body.32.lr.ph
  %indvars.iv136 = phi i64 [ 0, %for.body.32.lr.ph ], [ %indvars.iv.next137, %for.end.54 ]
  br i1 false, label %for.end.54, label %for.body.40.lr.ph

for.body.40.lr.ph:                                ; preds = %for.body.32
  br label %for.body.40

for.body.40:                                      ; preds = %for.body.40, %for.body.40.lr.ph
  %indvars.iv.next129 = add nuw nsw i64 0, 1
  %lftr.wideiv132 = trunc i64 %indvars.iv.next129 to i32
  br i1 false, label %for.body.40, label %for.cond.38.for.end.54_crit_edge

for.cond.38.for.end.54_crit_edge:                 ; preds = %for.body.40
  br label %for.end.54

for.end.54:                                       ; preds = %for.cond.38.for.end.54_crit_edge, %for.body.32
  %indvars.iv.next137 = add nuw nsw i64 %indvars.iv136, 1
  %lftr.wideiv138 = trunc i64 %indvars.iv.next137 to i32
  br i1 false, label %for.body.32, label %for.cond.30.for.cond.loopexit_crit_edge

for.end.65:                                       ; preds = %for.cond.loopexit
  %tmp1 = bitcast double* %b to i64*
  %tmp2 = load i64, i64* %tmp1, align 8, !tbaa !1
  %tmp3 = bitcast double* %y to i64*
  store i64 %tmp2, i64* %tmp3, align 8, !tbaa !1
  br label %for.body.70

for.body.70:                                      ; preds = %for.end.86, %for.end.65
  %arrayidx72 = getelementptr inbounds double, double* %b, i64 0
  %tmp4 = load double, double* %arrayidx72, align 8, !tbaa !1
  br i1 true, label %for.body.75.lr.ph, label %for.end.86

for.body.75.lr.ph:                                ; preds = %for.body.70
  br label %for.body.75

for.body.75:                                      ; preds = %for.body.75, %for.body.75.lr.ph
  %w.284 = phi double [ %tmp4, %for.body.75.lr.ph ], [ %sub83, %for.body.75 ]
  %sub83 = fsub double %w.284, undef
  br i1 false, label %for.body.75, label %for.cond.73.for.end.86_crit_edge

for.cond.73.for.end.86_crit_edge:                 ; preds = %for.body.75
  br label %for.end.86

for.end.86:                                       ; preds = %for.cond.73.for.end.86_crit_edge, %for.body.70
  br i1 false, label %for.body.70, label %for.end.91

for.end.91:                                       ; preds = %for.end.86
  br label %for.body.99

for.body.99:                                      ; preds = %for.end.118, %for.end.91
  br i1 true, label %for.body.106.lr.ph, label %for.end.118

for.body.106.lr.ph:                               ; preds = %for.body.99
  br label %for.body.106

for.body.106:                                     ; preds = %for.body.106, %for.body.106.lr.ph
  br i1 undef, label %for.body.106, label %for.cond.104.for.end.118_crit_edge

for.cond.104.for.end.118_crit_edge:               ; preds = %for.body.106
  br label %for.end.118

for.end.118:                                      ; preds = %for.cond.104.for.end.118_crit_edge, %for.body.99
  br i1 undef, label %for.body.99, label %for.end.131

for.end.131:                                      ; preds = %for.end.118
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
