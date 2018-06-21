; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Simple 3-pair chain with loads and stores
; CHECK: test1
; CHECK: store <2 x double>
; CHECK: ret
define void @test1(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
}

; Simple 3-pair chain with loads and stores, obfuscated with bitcasts
; CHECK: test2
; CHECK: store <2 x double>
; CHECK: ret
define void @test2(double* %a, double* %b, i8* %e) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  %c = bitcast i8* %e to double*
  store double %mul, double* %c, align 8
  %carrayidx5 = getelementptr inbounds i8, i8* %e, i64 8
  %arrayidx5 = bitcast i8* %carrayidx5 to double*
  store double %mul5, double* %arrayidx5, align 8
  ret void
}

; Don't vectorize volatile loads.
; CHECK: test_volatile_load
; CHECK-NOT: load <2 x double>
; CHECK: store <2 x double>
; CHECK: ret
define void @test_volatile_load(double* %a, double* %b, double* %c) {
entry:
  %i0 = load volatile double, double* %a, align 8
  %i1 = load volatile double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
}

; Don't vectorize volatile stores.
; CHECK: test_volatile_store
; CHECK-NOT: store <2 x double>
; CHECK: ret
define void @test_volatile_store(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double, double* %a, align 8
  %i1 = load double, double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double, double* %a, i64 1
  %i3 = load double, double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double, double* %b, i64 1
  %i4 = load double, double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store volatile double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double, double* %c, i64 1
  store volatile double %mul5, double* %arrayidx5, align 8
  ret void
}


