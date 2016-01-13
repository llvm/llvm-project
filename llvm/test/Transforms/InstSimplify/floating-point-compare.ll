; RUN: opt < %s -instsimplify -S | FileCheck %s

; These tests choose arbitrarily between float and double,
; and between uge and olt, to give reasonble coverage 
; without combinatorial explosion.

declare float @llvm.fabs.f32(float)
declare float @llvm.sqrt.f32(float)
declare double @llvm.powi.f64(double,i32)
declare float @llvm.exp.f32(float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.exp2.f64(double)
declare float @llvm.fma.f32(float,float,float)

declare void @expect_equal(i1,i1)

; CHECK-LABEL: @orderedLessZeroTree(
define i1 @orderedLessZeroTree(float,float,float,float) {
  %square = fmul float %0, %0
  %abs = call float @llvm.fabs.f32(float %1)
  %sqrt = call float @llvm.sqrt.f32(float %2)
  %fma = call float @llvm.fma.f32(float %3, float %3, float %sqrt)
  %div = fdiv float %square, %abs
  %rem = frem float %sqrt, %fma
  %add = fadd float %div, %rem
  %uge = fcmp uge float %add, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

; CHECK-LABEL: @orderedLessZeroExpExt(
define i1 @orderedLessZeroExpExt(float) {
  %a = call float @llvm.exp.f32(float %0)
  %b = fpext float %a to double
  %uge = fcmp uge double %b, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

; CHECK-LABEL: @orderedLessZeroExp2Trunc(
define i1 @orderedLessZeroExp2Trunc(double) {
  %a = call double @llvm.exp2.f64(double %0)
  %b = fptrunc double %a to float
  %olt = fcmp olt float %b, 0.000000e+00
; CHECK: ret i1 false
  ret i1 %olt
}

; CHECK-LABEL: @orderedLessZeroPowi(
define i1 @orderedLessZeroPowi(double,double) {
  ; Even constant exponent
  %a = call double @llvm.powi.f64(double %0, i32 2)
  %square = fmul double %1, %1
  ; Odd constant exponent with provably non-negative base
  %b = call double @llvm.powi.f64(double %square, i32 3)
  %c = fadd double %a, %b
  %olt = fcmp olt double %b, 0.000000e+00
; CHECK: ret i1 false
  ret i1 %olt
}

; CHECK-LABEL: @orderedLessZeroUIToFP(
define i1 @orderedLessZeroUIToFP(i32) {
  %a = uitofp i32 %0 to float
  %uge = fcmp uge float %a, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

; CHECK-LABEL: @orderedLessZeroSelect(
define i1 @orderedLessZeroSelect(float, float) {
  %a = call float @llvm.exp.f32(float %0)
  %b = call float @llvm.fabs.f32(float %1)
  %c = fcmp olt float %0, %1
  %d = select i1 %c, float %a, float %b
  %e = fadd float %d, 1.0
  %uge = fcmp uge float %e, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

; CHECK-LABEL: @orderedLessZeroMinNum(
define i1 @orderedLessZeroMinNum(float, float) {
  %a = call float @llvm.exp.f32(float %0)
  %b = call float @llvm.fabs.f32(float %1)
  %c = call float @llvm.minnum.f32(float %a, float %b)
  %uge = fcmp uge float %c, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

; CHECK-LABEL: @orderedLessZeroMaxNum(
define i1 @orderedLessZeroMaxNum(float, float) {
  %a = call float @llvm.exp.f32(float %0)
  %b = call float @llvm.maxnum.f32(float %a, float %1)
  %uge = fcmp uge float %b, 0.000000e+00
; CHECK: ret i1 true
  ret i1 %uge
}

define i1 @nonans1(double %in1, double %in2) {
  %cmp = fcmp nnan uno double %in1, %in2
  ret i1 %cmp

; CHECK-LABEL: @nonans1
; CHECK-NEXT: ret i1 false
}

define i1 @nonans2(double %in1, double %in2) {
  %cmp = fcmp nnan ord double %in1, %in2
  ret i1 %cmp

; CHECK-LABEL: @nonans2
; CHECK-NEXT: ret i1 true
}
