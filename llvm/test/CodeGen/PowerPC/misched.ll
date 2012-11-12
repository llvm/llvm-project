; RUN: llc < %s -enable-misched -verify-machineinstrs
; PR14302
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-bgq-linux"

@b = external global [16000 x double], align 32

define void @pr14302() nounwind {
entry:
  tail call void @putchar() nounwind
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.body24.i

for.body24.i:                                     ; preds = %for.body24.i, %for.body
  store double 1.000000e+00, double* undef, align 8
  br i1 undef, label %for.body24.i58, label %for.body24.i

for.body24.i58:                                   ; preds = %for.body24.i58, %for.body24.i
  %arrayidx26.i55.1 = getelementptr inbounds [16000 x double]* @b, i64 0, i64 undef
  store double 1.000000e+00, double* %arrayidx26.i55.1, align 8
  br i1 undef, label %for.body24.i64, label %for.body24.i58

for.body24.i64:                                   ; preds = %for.body24.i64, %for.body24.i58
  %exitcond.2489 = icmp eq i32 0, 16000
  br i1 %exitcond.2489, label %for.body24.i70, label %for.body24.i64

for.body24.i70:                                   ; preds = %for.body24.i70, %for.body24.i64
  br i1 undef, label %for.body24.i76, label %for.body24.i70

for.body24.i76:                                   ; preds = %for.body24.i76, %for.body24.i70
  br i1 undef, label %set1d.exit77, label %for.body24.i76

set1d.exit77:                                     ; preds = %for.body24.i76
  br label %for.body29

for.body29:                                       ; preds = %for.body29, %set1d.exit77
  br i1 undef, label %for.end35, label %for.body29

for.end35:                                        ; preds = %for.body29
  ret void
}

declare void @putchar()
