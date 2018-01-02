; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

define zeroext i1 @fcmp_float1(float %a) {
; CHECK-LABEL: fcmp_float1
; CHECK:       fcmp s0, #0.0
; CHECK-NEXT:  cset {{w[0-9]+}}, ne
  %1 = fcmp une float %a, 0.000000e+00
  ret i1 %1
}

define zeroext i1 @fcmp_float2(float %a, float %b) {
; CHECK-LABEL: fcmp_float2
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, ne
  %1 = fcmp une float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_double1(double %a) {
; CHECK-LABEL: fcmp_double1
; CHECK:       fcmp d0, #0.0
; CHECK-NEXT:  cset {{w[0-9]+}}, ne
  %1 = fcmp une double %a, 0.000000e+00
  ret i1 %1
}

define zeroext i1 @fcmp_double2(double %a, double %b) {
; CHECK-LABEL: fcmp_double2
; CHECK:       fcmp d0, d1
; CHECK-NEXT:  cset {{w[0-9]+}}, ne
  %1 = fcmp une double %a, %b
  ret i1 %1
}

; Check each fcmp condition
define zeroext i1 @fcmp_false(float %a) {
; CHECK-LABEL: fcmp_false
; CHECK:       mov {{w[0-9]+}}, wzr
  %1 = fcmp ogt float %a, %a
  ret i1 %1
}

define zeroext i1 @fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: fcmp_oeq
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, eq
  %1 = fcmp oeq float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ogt(float %a, float %b) {
; CHECK-LABEL: fcmp_ogt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, gt
  %1 = fcmp ogt float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_oge(float %a, float %b) {
; CHECK-LABEL: fcmp_oge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, ge
  %1 = fcmp oge float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_olt(float %a, float %b) {
; CHECK-LABEL: fcmp_olt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, mi
  %1 = fcmp olt float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ole(float %a, float %b) {
; CHECK-LABEL: fcmp_ole
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, ls
  %1 = fcmp ole float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_one(float %a, float %b) {
; CHECK-LABEL: fcmp_one
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset [[REG:w[0-9]+]], mi
; CHECK-NEXT:  csinc {{w[0-9]+}}, [[REG]], wzr, le
  %1 = fcmp one float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ord(float %a, float %b) {
; CHECK-LABEL: fcmp_ord
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, vc
  %1 = fcmp ord float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_uno(float %a, float %b) {
; CHECK-LABEL: fcmp_uno
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, vs
  %1 = fcmp uno float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ueq(float %a, float %b) {
; CHECK-LABEL: fcmp_ueq
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset [[REG:w[0-9]+]], eq
; CHECK-NEXT:  csinc {{w[0-9]+}}, [[REG]], wzr, vc
  %1 = fcmp ueq float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ugt(float %a, float %b) {
; CHECK-LABEL: fcmp_ugt
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, hi
  %1 = fcmp ugt float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_uge(float %a, float %b) {
; CHECK-LABEL: fcmp_uge
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, pl
  %1 = fcmp uge float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ult(float %a, float %b) {
; CHECK-LABEL: fcmp_ult
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, lt
  %1 = fcmp ult float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_ule(float %a, float %b) {
; CHECK-LABEL: fcmp_ule
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, le
  %1 = fcmp ule float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_une(float %a, float %b) {
; CHECK-LABEL: fcmp_une
; CHECK:       fcmp s0, s1
; CHECK-NEXT:  cset {{w[0-9]+}}, ne
  %1 = fcmp une float %a, %b
  ret i1 %1
}

define zeroext i1 @fcmp_true(float %a) {
; CHECK-LABEL: fcmp_true
; CHECK:       orr {{w[0-9]+}}, wzr, #0x1
  %1 = fcmp ueq float %a, %a
  ret i1 %1
}
