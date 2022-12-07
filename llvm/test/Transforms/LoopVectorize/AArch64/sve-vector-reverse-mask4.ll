; This is the loop in c++ being vectorize in this file with
; experimental.vector.reverse

;#pragma clang loop vectorize_width(4, scalable)
;  for (long int i = N - 1; i >= 0; i--)
;  {
;    if (cond[i])
;      a[i] += 1;
;  }

; The test checks if the mask is being correctly created, reverted and used

; RUN: opt -passes=loop-vectorize,dce,instcombine -mtriple aarch64-linux-gnu -S \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @vector_reverse_mask_nxv4i1(double* %a, double* %cond, i64 %N) #0 {
; CHECK-LABEL: vector.body:
; CHECK: %[[REVERSE6:.*]] = call <vscale x 4 x i1> @llvm.experimental.vector.reverse.nxv4i1(<vscale x 4 x i1> %{{.*}})
; CHECK: %[[WIDEMSKLOAD:.*]] = call <vscale x 4 x double> @llvm.masked.load.nxv4f64.p0nxv4f64(<vscale x 4 x double>* %{{.*}}, i32 8, <vscale x 4 x i1> %[[REVERSE6]], <vscale x 4 x double> poison)
; CHECK-NEXT: %[[FADD:.*]] = fadd <vscale x 4 x double> %[[WIDEMSKLOAD]]
; CHECK:  %[[REVERSE9:.*]] = call <vscale x 4 x i1> @llvm.experimental.vector.reverse.nxv4i1(<vscale x 4 x i1> %{{.*}})
; CHECK: call void @llvm.masked.store.nxv4f64.p0nxv4f64(<vscale x 4 x double> %[[FADD]], <vscale x 4 x double>* %{{.*}}, i32 8, <vscale x 4 x i1> %[[REVERSE9]]

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
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
