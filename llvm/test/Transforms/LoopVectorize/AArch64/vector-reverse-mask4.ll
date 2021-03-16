; This is the loop in c++ being vectorize in this file with 
; shuffle reverse

;#pragma clang loop vectorize_width(4, fixed)
;  for (long int i = N - 1; i >= 0; i--)
;  {
;    if (cond[i])
;      a[i] += 1;
;  }

; The test checks if the mask is being correctly created, reverted  and used

; RUN: opt -loop-vectorize -dce -instcombine -mtriple aarch64-linux-gnu -S < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning


target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @vector_reverse_mask_v4i1(double* %a, double* %cond, i64 %N) #0 {
; CHECK-LABEL: vector.body:
; CHECK: %[[REVERSE6:.*]] = shufflevector <4 x i1> %{{.*}}, <4 x i1> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK: %[[WIDEMSKLOAD:.*]] = call <4 x double> @llvm.masked.load.v4f64.p0v4f64(<4 x double>* nonnull %{{.*}}, i32 8, <4 x i1> %[[REVERSE6]], <4 x double> poison)
; CHECK-NEXT: %[[FADD:.*]] = fadd <4 x double> %[[WIDEMSKLOAD]]
; CHECK: call void @llvm.masked.store.v4f64.p0v4f64(<4 x double> %[[FADD]], <4 x double>* %{{.*}}, i32 8, <4 x i1> %[[REVERSE6]])

entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup, %entry
  ret void

for.body:                                         ; preds = %for.body, %entry
  %i.08.in = phi i64 [ %i.08, %for.inc ], [ %N, %entry ]
  %i.08 = add nsw i64 %i.08.in, -1
  %arrayidx = getelementptr inbounds double, double* %cond, i64 %i.08
  %0 = load double, double* %arrayidx, align 8
  %tobool = fcmp une double %0, 0.000000e+00
  br i1 %tobool, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %arrayidx1 = getelementptr inbounds double, double* %a, i64 %i.08
  %1 = load double, double* %arrayidx1, align 8
  %add = fadd double %1, 1.000000e+00
  store double %add, double* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %cmp = icmp sgt i64 %i.08.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

attributes #0 = {"target-cpu"="generic" "target-features"="+neon,+sve"}


!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
