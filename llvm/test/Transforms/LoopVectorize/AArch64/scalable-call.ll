; RUN: opt -S -loop-vectorize -force-vector-interleave=1 -instcombine -mattr=+sve -mtriple aarch64-unknown-linux-gnu < %s | FileCheck %s

define void @vec_load(i64 %N, double* nocapture %a, double* nocapture readonly %b) {
; CHECK-LABEL: @vec_load
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x double>, <vscale x 2 x double>*
; CHECK: call <vscale x 2 x double> @foo_vec(<vscale x 2 x double> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %b, i64 %iv
  %0 = load double, double* %arrayidx, align 8
  %1 = call double @foo(double %0) #0
  %add = fadd double %1, 1.000000e+00
  %arrayidx2 = getelementptr inbounds double, double* %a, i64 %iv
  store double %add, double* %arrayidx2, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:                                 ; preds = %for.body, %entry
  ret void
}

define void @vec_scalar(i64 %N, double* nocapture %a) {
; CHECK-LABEL: @vec_scalar
; CHECK: vector.body:
; CHECK: call <vscale x 2 x double> @foo_vec(<vscale x 2 x double> shufflevector (<vscale x 2 x double> insertelement (<vscale x 2 x double> poison, double 1.000000e+01, i32 0), <vscale x 2 x double> poison, <vscale x 2 x i32> zeroinitializer))
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %0 = call double @foo(double 10.0) #0
  %sub = fsub double %0, 1.000000e+00
  %arrayidx = getelementptr inbounds double, double* %a, i64 %iv
  store double %sub, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:                                 ; preds = %for.body, %entry
  ret void
}

define void @vec_ptr(i64 %N, i64* noalias %a, i64** readnone %b) {
; CHECK-LABEL: @vec_ptr
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x i64*>, <vscale x 2 x i64*>*
; CHECK: call <vscale x 2 x i64> @bar_vec(<vscale x 2 x i64*> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %gep = getelementptr i64*, i64** %b, i64 %iv
  %load = load i64*, i64** %gep
  %call = call i64 @bar(i64* %load) #1
  %arrayidx = getelementptr inbounds i64, i64* %a, i64 %iv
  store i64 %call, i64* %arrayidx
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

define void @vec_intrinsic(i64 %N, double* nocapture readonly %a) {
;; FIXME: Should be calling sin_vec, once the cost of scalarizing is handled.
; CHECK-LABEL: @vec_intrinsic
; CHECK: vector.body:
; CHECK: %[[LOAD:.*]] = load <vscale x 2 x double>, <vscale x 2 x double>*
; CHECK: call fast <vscale x 2 x double> @llvm.sin.nxv2f64(<vscale x 2 x double> %[[LOAD]])
entry:
  %cmp7 = icmp sgt i64 %N, 0
  br i1 %cmp7, label %for.body, label %for.end

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %a, i64 %iv
  %0 = load double, double* %arrayidx, align 8
  %1 = call fast double @llvm.sin.f64(double %0) #2
  %add = fadd fast double %1, 1.000000e+00
  store double %add, double* %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !1

for.end:
  ret void
}

declare double @foo(double)
declare i64 @bar(i64*)
declare double @llvm.sin.f64(double)

declare <vscale x 2 x double> @foo_vec(<vscale x 2 x double>)
declare <vscale x 2 x i64> @bar_vec(<vscale x 2 x i64*>)
declare <vscale x 2 x double> @sin_vec(<vscale x 2 x double>)

attributes #0 = { "vector-function-abi-variant"="_ZGV_LLVM_Nxv_foo(foo_vec)" }
attributes #1 = { "vector-function-abi-variant"="_ZGV_LLVM_Nxv_bar(bar_vec)" }
attributes #2 = { "vector-function-abi-variant"="_ZGV_LLVM_Nxv_llvm.sin.f64(sin_vec)" }

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 2}
!3 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
