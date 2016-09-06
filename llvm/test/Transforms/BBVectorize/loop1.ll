target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-ignore-target-info -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -basicaa -loop-unroll -unroll-threshold=45 -unroll-allow-partial -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-ignore-target-info -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-UNRL
; The second check covers the use of alias analysis (with loop unrolling).

define void @test1(double* noalias %out, double* noalias %in1, double* noalias %in2) nounwind uwtable {
entry:
  br label %for.body
; CHECK-LABEL: @test1(
; CHECK-UNRL-LABEL: @test1(

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %in1, i64 %indvars.iv
  %0 = load double, double* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds double, double* %in2, i64 %indvars.iv
  %1 = load double, double* %arrayidx2, align 8
  %mul = fmul double %0, %0
  %mul3 = fmul double %0, %1
  %add = fadd double %mul, %mul3
  %add4 = fadd double %1, %1
  %add5 = fadd double %add4, %0
  %mul6 = fmul double %0, %add5
  %add7 = fadd double %add, %mul6
  %mul8 = fmul double %1, %1
  %add9 = fadd double %0, %0
  %add10 = fadd double %add9, %0
  %mul11 = fmul double %mul8, %add10
  %add12 = fadd double %add7, %mul11
  %arrayidx14 = getelementptr inbounds double, double* %out, i64 %indvars.iv
  store double %add12, double* %arrayidx14, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10
  br i1 %exitcond, label %for.end, label %for.body
; CHECK: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK: %arrayidx = getelementptr inbounds double, double* %in1, i64 %indvars.iv
; CHECK: %0 = load double, double* %arrayidx, align 8
; CHECK: %arrayidx2 = getelementptr inbounds double, double* %in2, i64 %indvars.iv
; CHECK: %1 = load double, double* %arrayidx2, align 8
; CHECK: %mul = fmul double %0, %0
; CHECK: %mul3 = fmul double %0, %1
; CHECK: %add = fadd double %mul, %mul3
; CHECK: %mul8 = fmul double %1, %1
; CHECK: %add4.v.i1.1 = insertelement <2 x double> undef, double %1, i32 0
; CHECK: %add4.v.i1.2 = insertelement <2 x double> %add4.v.i1.1, double %0, i32 1
; CHECK: %add4 = fadd <2 x double> %add4.v.i1.2, %add4.v.i1.2
; CHECK: %add5.v.i1.1 = insertelement <2 x double> undef, double %0, i32 0
; CHECK: %add5.v.i1.2 = insertelement <2 x double> %add5.v.i1.1, double %0, i32 1
; CHECK: %add5 = fadd <2 x double> %add4, %add5.v.i1.2
; CHECK: %mul6.v.i0.2 = insertelement <2 x double> %add5.v.i1.1, double %mul8, i32 1
; CHECK: %mul6 = fmul <2 x double> %mul6.v.i0.2, %add5
; CHECK: %mul6.v.r1 = extractelement <2 x double> %mul6, i32 0
; CHECK: %mul6.v.r2 = extractelement <2 x double> %mul6, i32 1
; CHECK: %add7 = fadd double %add, %mul6.v.r1
; CHECK: %add12 = fadd double %add7, %mul6.v.r2
; CHECK: %arrayidx14 = getelementptr inbounds double, double* %out, i64 %indvars.iv
; CHECK: store double %add12, double* %arrayidx14, align 8
; CHECK: %indvars.iv.next = add i64 %indvars.iv, 1
; CHECK: %lftr.wideiv = trunc i64 %indvars.iv.next to i32
; CHECK: %exitcond = icmp eq i32 %lftr.wideiv, 10
; CHECK: br i1 %exitcond, label %for.end, label %for.body
; CHECK-UNRL: %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next.1, %for.body ]
; CHECK-UNRL: %arrayidx = getelementptr inbounds double, double* %in1, i64 %indvars.iv
; CHECK-UNRL: %0 = bitcast double* %arrayidx to <2 x double>*
; CHECK-UNRL: %arrayidx2 = getelementptr inbounds double, double* %in2, i64 %indvars.iv
; CHECK-UNRL: %1 = bitcast double* %arrayidx2 to <2 x double>*
; CHECK-UNRL: %arrayidx14 = getelementptr inbounds double, double* %out, i64 %indvars.iv
; CHECK-UNRL: %2 = load <2 x double>, <2 x double>* %0, align 8
; CHECK-UNRL: %3 = load <2 x double>, <2 x double>* %1, align 8
; CHECK-UNRL: %mul = fmul <2 x double> %2, %2
; CHECK-UNRL: %mul3 = fmul <2 x double> %2, %3
; CHECK-UNRL: %add = fadd <2 x double> %mul, %mul3
; CHECK-UNRL: %add4 = fadd <2 x double> %3, %3
; CHECK-UNRL: %add5 = fadd <2 x double> %add4, %2
; CHECK-UNRL: %mul6 = fmul <2 x double> %2, %add5
; CHECK-UNRL: %add7 = fadd <2 x double> %add, %mul6
; CHECK-UNRL: %mul8 = fmul <2 x double> %3, %3
; CHECK-UNRL: %add9 = fadd <2 x double> %2, %2
; CHECK-UNRL: %add10 = fadd <2 x double> %add9, %2
; CHECK-UNRL: %mul11 = fmul <2 x double> %mul8, %add10
; CHECK-UNRL: %add12 = fadd <2 x double> %add7, %mul11
; CHECK-UNRL: %4 = bitcast double* %arrayidx14 to <2 x double>*
; CHECK-UNRL: store <2 x double> %add12, <2 x double>* %4, align 8
; CHECK-UNRL: %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
; CHECK-UNRL: %lftr.wideiv.1 = trunc i64 %indvars.iv.next.1 to i32
; CHECK-UNRL: %exitcond.1 = icmp eq i32 %lftr.wideiv.1, 10
; CHECK-UNRL: br i1 %exitcond.1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
