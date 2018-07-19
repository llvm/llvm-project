; RUN: llc < %s

; This used to assert with "Overran sorted position" in AssignTopologicalOrder
; due to a cycle created in performPostLD1Combine.

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

; Function Attrs: nounwind ssp
define void @f(double* %P1) #0 {
entry:
  %arrayidx4 = getelementptr inbounds double, double* %P1, i64 1
  %0 = load double, double* %arrayidx4, align 8, !tbaa !1
  %1 = load double, double* %P1, align 8, !tbaa !1
  %2 = insertelement <2 x double> undef, double %0, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  %4 = fsub <2 x double> zeroinitializer, %3
  %5 = fmul <2 x double> undef, %4
  %6 = extractelement <2 x double> %5, i32 0
  %cmp168 = fcmp olt double %6, undef
  br i1 %cmp168, label %if.then172, label %return

if.then172:                                       ; preds = %cond.end90
  %7 = tail call i64 @llvm.objectsize.i64.p0i8(i8* undef, i1 false)
  br label %return

return:                                           ; preds = %if.then172, %cond.end90, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) #1

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
