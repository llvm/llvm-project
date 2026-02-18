; RUN: opt -S -passes=instcombine -o - %s | FileCheck %s

; Test that fast math lib call simplification of double math function to float
; equivalent doesn't occur when the calling function matches the float
; equivalent math function. Otherwise this can cause the generation of infinite
; loops when compiled with -O2/3 and fast math.

; Test case C source:
;
;   extern double exp(double x);
;   inline float expf(float x) { return (float) exp((double) x); }
;   float fn(float f) { return expf(f); }
;
; IR generated with command:
;
;   clang -cc1 -O2 -ffast-math -emit-llvm -disable-llvm-passes -triple x86_64-unknown-unknown -o - <srcfile>

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Function Attrs: nounwind
define float @fn(float %f) {
; CHECK: define float @fn(
; CHECK: call fast float @expf(
  %f.addr = alloca float, align 4
  store float %f, ptr %f.addr, align 4, !tbaa !1
  %1 = load float, ptr %f.addr, align 4, !tbaa !1
  %call = call fast float @expf(float %1)
  ret float %call
}

; Function Attrs: inlinehint nounwind readnone
define available_externally float @expf(float %x) {
; CHECK: define available_externally float @expf(
; CHECK: fpext float
; CHECK: call fast double @exp(
; CHECK: fptrunc double
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4, !tbaa !1
  %1 = load float, ptr %x.addr, align 4, !tbaa !1
  %conv = fpext float %1 to double
  %call = call fast double @exp(double %conv)
  %conv1 = fptrunc double %call to float
  ret float %conv1
}

; Function Attrs: nounwind readnone
declare double @exp(double)

!llvm.ident = !{!0}

!0 = !{!"clang version 5.0.0"}
!1 = !{!2, !2, i64 0}
!2 = !{!"float", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
