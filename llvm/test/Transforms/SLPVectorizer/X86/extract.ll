; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"
;CHECK: fextr
;CHECK-NOT: insertelement
;CHECK-NOT: extractelement
;CHECK: fadd <2 x double>
;CHECK: ret void
define void @fextr(double* %ptr) {
entry:
  %LD = load <2 x double>, <2 x double>* undef
  %V0 = extractelement <2 x double> %LD, i32 0
  %V1 = extractelement <2 x double> %LD, i32 1
  %P0 = getelementptr inbounds double, double* %ptr, i64 0
  %P1 = getelementptr inbounds double, double* %ptr, i64 1
  %A0 = fadd double %V0, 0.0
  %A1 = fadd double %V1, 1.1
  store double %A0, double* %P0, align 4
  store double %A1, double* %P1, align 4
  ret void
}

;CHECK: fextr1
;CHECK: insertelement
;CHECK: insertelement
;CHECK: ret void
define void @fextr1(double* %ptr) {
entry:
  %LD = load <2 x double>, <2 x double>* undef
  %V0 = extractelement <2 x double> %LD, i32 0
  %V1 = extractelement <2 x double> %LD, i32 1
  %P0 = getelementptr inbounds double, double* %ptr, i64 1  ; <--- incorrect order
  %P1 = getelementptr inbounds double, double* %ptr, i64 0
  %A0 = fadd double %V0, 1.2
  %A1 = fadd double %V1, 3.4
  store double %A0, double* %P0, align 4
  store double %A1, double* %P1, align 4
  ret void
}

;CHECK: fextr2
;CHECK: insertelement
;CHECK: insertelement
;CHECK: ret void
define void @fextr2(double* %ptr) {
entry:
  %LD = load <4 x double>, <4 x double>* undef
  %V0 = extractelement <4 x double> %LD, i32 0  ; <--- invalid size.
  %V1 = extractelement <4 x double> %LD, i32 1
  %P0 = getelementptr inbounds double, double* %ptr, i64 0
  %P1 = getelementptr inbounds double, double* %ptr, i64 1
  %A0 = fadd double %V0, 5.5
  %A1 = fadd double %V1, 6.6
  store double %A0, double* %P0, align 4
  store double %A1, double* %P1, align 4
  ret void
}

