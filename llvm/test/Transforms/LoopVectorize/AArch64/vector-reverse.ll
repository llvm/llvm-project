; Test VLA for reverse with fixed size vector
; This is the loop in c++ being vectorize in this file with
; shuffle reverse
;  #pragma clang loop vectorize_width(8, fixed)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; RUN: opt -loop-vectorize -dce  -mtriple aarch64-linux-gnu -S < %s 2>%t | FileCheck %s

; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.$
; WARN-NOT: warning

define void @vector_reverse_f64(i64 %N, double* %a, double* %b) #0 {
; CHECK-LABEL: vector_reverse_f64
; CHECK-LABEL: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds double, double* %{{.*}}, i32 0
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds double, double* %[[GEP]], i32 -7
; CHECK-NEXT: %[[CAST:.*]] = bitcast double* %[[GEP1]] to <8 x double>*
; CHECK-NEXT: %[[WIDE:.*]] = load <8 x double>, <8 x double>* %[[CAST]], align 8
; CHECK-NEXT: %[[REVERSE:.*]] = shufflevector <8 x double> %[[WIDE]], <8 x double> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: %[[FADD:.*]] = fadd <8 x double> %[[REVERSE]]
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds double, double* {{.*}}, i64 {{.*}}
; CHECK-NEXT: %[[REVERSE6:.*]] = shufflevector <8 x double> %[[FADD]], <8 x double> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds double, double* %[[GEP2]], i32 0
; CHECK-NEXT: %[[GEP4:.*]] = getelementptr inbounds double, double* %[[GEP3]], i32 -7
; CHECK-NEXT: %[[CAST:.*]] = bitcast double* %[[GEP4]] to <8 x double>*
; CHECK-NEXT:  store <8 x double> %[[REVERSE6]], <8 x double>* %[[CAST]], align 8

entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.08.in = phi i64 [ %i.08, %for.body ], [ %N, %entry ]
  %i.08 = add nsw i64 %i.08.in, -1
  %arrayidx = getelementptr inbounds double, double* %b, i64 %i.08
  %0 = load double, double* %arrayidx, align 8
  %add = fadd double %0, 1.000000e+00
  %arrayidx1 = getelementptr inbounds double, double* %a, i64 %i.08
  store double %add, double* %arrayidx1, align 8
  %cmp = icmp sgt i64 %i.08.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

define void @vector_reverse_i64(i64 %N, i64* %a, i64* %b) #0 {
; CHECK-LABEL: vector_reverse_i64
; CHECK-LABEL: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds i64, i64* %{{.*}}, i32 0
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds i64, i64* %[[GEP]], i32 -7
; CHECK-NEXT: %[[CAST:.*]] = bitcast i64* %[[GEP1]] to <8 x i64>*
; CHECK-NEXT: %[[WIDE:.*]] = load <8 x i64>, <8 x i64>* %[[CAST]], align 8
; CHECK-NEXT: %[[REVERSE:.*]] = shufflevector <8 x i64> %[[WIDE]], <8 x i64> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: %[[FADD:.*]] = add <8 x i64> %[[REVERSE]]
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds i64, i64* {{.*}}, i64 {{.*}}
; CHECK-NEXT: %[[REVERSE6:.*]] = shufflevector <8 x i64> %[[FADD]], <8 x i64> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds i64, i64* %[[GEP2]], i32 0
; CHECK-NEXT: %[[GEP4:.*]] = getelementptr inbounds i64, i64* %[[GEP3]], i32 -7
; CHECK-NEXT: %[[CAST1:.*]] = bitcast i64* %[[GEP4]] to <8 x i64>*
; CHECK-NEXT:  store <8 x i64> %[[REVERSE6]], <8 x i64>* %[[CAST1]], align 8

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup, %entry
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.09.in = phi i64 [ %i.09, %for.body ], [ %N, %entry ]
  %i.09 = add nsw i64 %i.09.in, -1
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %i.09
  %0 = load i64, i64* %arrayidx, align 8
  %add = add i64 %0, 1
  %arrayidx2 = getelementptr inbounds i64, i64* %a, i64 %i.09
  store i64 %add, i64* %arrayidx2, align 8
  %cmp = icmp sgt i64 %i.09.in, 1
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !0
}

attributes #0 = { "target-cpu"="generic" "target-features"="+neon,+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.mustprogress"}
!2 = !{!"llvm.loop.vectorize.width", i32 8}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 false}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
