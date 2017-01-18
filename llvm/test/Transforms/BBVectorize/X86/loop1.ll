target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -basicaa -loop-unroll -unroll-partial-threshold=45 -unroll-allow-partial -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-UNRL
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
; CHECK: insertelement
; CHECK-NEXT: insertelement
; CHECK-NEXT: fadd <2 x double>
; CHECK-NEXT: insertelement
; CHECK-NEXT: shufflevector
; CHECK-NEXT: fadd <2 x double>
; CHECK-NEXT: insertelement
; CHECK-NEXT: fmul <2 x double>

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

for.end:                                          ; preds = %for.body
  ret void
}
