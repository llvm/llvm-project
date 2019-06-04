; REQUIRES: asserts
; RUN: llc -mtriple=x86_64-unknown-unknown -debug-only=isel < %s -o /dev/null 2>&1 | FileCheck %s

; This tests the propagation of fast-math-flags from IR instructions to SDNodeFlags.

; CHECK-LABEL: Initial selection DAG: %bb.0 'fmf_transfer:'

; CHECK:         t5: f32 = fadd nsz t2, t4
; CHECK-NEXT:    t6: f32 = fadd arcp t5, t4
; CHECK-NEXT:    t7: f32 = fadd nnan t6, t4
; CHECK-NEXT:    t8: f32 = fadd ninf t7, t4
; CHECK-NEXT:    t9: f32 = fadd contract t8, t4
; CHECK-NEXT:    t10: f32 = fadd afn t9, t4
; CHECK-NEXT:    t11: f32 = fadd reassoc t10, t4
; CHECK-NEXT:    t12: f32 = fadd nnan ninf nsz arcp contract afn reassoc t11, t4

; CHECK: Optimized lowered selection DAG: %bb.0 'fmf_transfer:'

define float @fmf_transfer(float %x, float %y) {
  %f1 = fadd nsz float %x, %y
  %f2 = fadd arcp float %f1, %y
  %f3 = fadd nnan float %f2, %y
  %f4 = fadd ninf float %f3, %y
  %f5 = fadd contract float %f4, %y
  %f6 = fadd afn float %f5, %y
  %f7 = fadd reassoc float %f6, %y
  %f8 = fadd fast float %f7, %y
  ret float %f8
}

; CHECK: Optimized type-legalized selection DAG: %bb.0 'fmf_setcc:'
; CHECK: t13: i8 = setcc nnan ninf nsz arcp contract afn reassoc t2, ConstantFP:f32<0.000000e+00>, setlt:ch

define float @fmf_setcc(float %x, float %y) {
  %cmp = fcmp fast ult float %x, 0.0
  %ret = select i1 %cmp, float %x, float %y
  ret float %ret
}
