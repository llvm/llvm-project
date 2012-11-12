target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=corei7 -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s

; Simple 3-pair chain with loads and stores
define void @test1(double* %a, double* %b, double* %c) nounwind uwtable readonly {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
; CHECK: @test1
; CHECK: %i0.v.i0 = bitcast double* %a to <2 x double>*
; CHECK: %i1.v.i0 = bitcast double* %b to <2 x double>*
; CHECK: %i0 = load <2 x double>* %i0.v.i0, align 8
; CHECK: %i1 = load <2 x double>* %i1.v.i0, align 8
; CHECK: %mul = fmul <2 x double> %i0, %i1
; CHECK: %0 = bitcast double* %c to <2 x double>*
; CHECK: store <2 x double> %mul, <2 x double>* %0, align 8
; CHECK: ret void
}

