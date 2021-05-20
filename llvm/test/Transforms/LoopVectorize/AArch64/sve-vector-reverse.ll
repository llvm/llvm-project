; This is the loop in c++ being vectorize in this file with
;experimental.vector.reverse
;  #pragma clang loop vectorize_width(8, scalable)
;  for (int i = N-1; i >= 0; --i)
;    a[i] = b[i] + 1.0;

; RUN: opt -loop-vectorize -scalable-vectorization=on -dce -instcombine -mtriple aarch64-linux-gnu -S < %s | FileCheck %s

define void @vector_reverse_f64(i64 %N, double* %a, double* %b) #0{
; CHECK-LABEL: @vector_reverse_f64
; CHECK-LABEL: vector.body:
; CHECK: %[[ADD:.*]] = add i64 %{{.*}}, %N
; CHECK-NEXT: %[[GEP:.*]] = getelementptr inbounds double, double* %b, i64 %[[ADD]]
; CHECK-NEXT: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[MUL:.*]] = mul i32 %[[VSCALE]], -8
; CHECK-NEXT: %[[OR:.*]] = or i32 %[[MUL]], 1
; CHECK-NEXT: %[[SEXT:.*]] = sext i32 %[[OR]] to i64
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds double, double* %[[GEP]], i64 %[[SEXT]]
; CHECK-NEXT: %[[CAST:.*]] = bitcast double* %[[GEP1]] to <vscale x 8 x double>*
; CHECK-NEXT: %[[WIDE:.*]] = load <vscale x 8 x double>, <vscale x 8 x double>* %[[CAST]], align 8
; CHECK-NEXT: %[[REVERSE:.*]] = call <vscale x 8 x double> @llvm.experimental.vector.reverse.nxv8f64(<vscale x 8 x double> %[[WIDE]])
; CHECK-NEXT: %[[FADD:.*]] = fadd <vscale x 8 x double> %[[REVERSE]], shufflevector
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds double, double* %a, i64 %[[ADD]]
; CHECK-NEXT: %[[REVERSE6:.*]] = call <vscale x 8 x double> @llvm.experimental.vector.reverse.nxv8f64(<vscale x 8 x double> %[[FADD]])
; CHECK-NEXT: %[[VSCALE1:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[MUL1:.*]] = mul i32 %[[VSCALE1]], -8
; CHECK-NEXT: %[[OR1:.*]] = or i32 %[[MUL1]], 1
; CHECK-NEXT: %[[SEXT1:.*]] = sext i32 %[[OR1]] to i64
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds double, double* %[[GEP2]], i64 %[[SEXT1]]
; CHECK-NEXT: %[[CAST1:.*]] = bitcast double* %[[GEP3]] to <vscale x 8 x double>*
; CHECK-NEXT: store <vscale x 8 x double> %[[REVERSE6]], <vscale x 8 x double>* %[[CAST1]], align 8

entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
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
; CHECK-LABEL: vector.body:
; CHECK: %[[ADD:.*]] = add i64 %{{.*}}, %N
; CHECK-NEXT: %[[GEP:.*]] = getelementptr inbounds i64, i64* %b, i64 %[[ADD]]
; CHECK-NEXT: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[MUL:.*]] = mul i32 %[[VSCALE]], -8
; CHECK-NEXT: %[[OR:.*]] = or i32 %[[MUL]], 1
; CHECK-NEXT: %[[SEXT:.*]] = sext i32 %[[OR]] to i64
; CHECK-NEXT: %[[GEP1:.*]] = getelementptr inbounds i64, i64* %[[GEP]], i64 %[[SEXT]]
; CHECK-NEXT: %[[CAST:.*]] = bitcast i64* %[[GEP1]] to <vscale x 8 x i64>*
; CHECK-NEXT: %[[WIDE:.*]] = load <vscale x 8 x i64>, <vscale x 8 x i64>* %[[CAST]], align 8
; CHECK-NEXT: %[[REVERSE:.*]] = call <vscale x 8 x i64> @llvm.experimental.vector.reverse.nxv8i64(<vscale x 8 x i64> %[[WIDE]])
; CHECK-NEXT: %[[ADD1:.*]] = add <vscale x 8 x i64> %[[REVERSE]]
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds i64, i64* %a, i64 %[[ADD]]
; CHECK-NEXT: %[[REVERSE6]] = call <vscale x 8 x i64> @llvm.experimental.vector.reverse.nxv8i64(<vscale x 8 x i64> %[[ADD1]])
; CHECK-NEXT: %[[VSCALE:.*]] = call i32 @llvm.vscale.i32()
; CHECK-NEXT: %[[MUL1:.*]] = mul i32 %[[VSCALE]], -8
; CHECK-NEXT: %[[OR1:.*]] = or i32 %[[MUL1]], 1
; CHECK-NEXT: %[[SEXT1:.*]] = sext i32 %[[OR1]] to i64
; CHECK-NEXT: %[[GEP3:.*]] = getelementptr inbounds i64, i64* %[[GEP2]], i64 %[[SEXT1]]
; CHECK-NEXT: %[[CAST1:.*]] = bitcast i64* %[[GEP3]] to <vscale x 8 x i64>*
; CHECK-NEXT:  store <vscale x 8 x i64> %[[REVERSE6]], <vscale x 8 x i64>* %[[CAST1]], align 8

entry:
  %cmp8 = icmp sgt i64 %N, 0
  br i1 %cmp8, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
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
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}

