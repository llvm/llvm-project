; Check that 0.0 is not materialized for CSEL when comparing against it.

; RUN: llc -mtriple=aarch64-linux-gnu -o - < %s | FileCheck %s

define float @foeq(float %a, float %b) #0 {
  %t = fcmp oeq float %a, 0.0
  %v = select i1 %t, float 0.0, float %b
  ret float %v
; CHECK-LABEL: foeq
; CHECK: fcmp [[R:s[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{s[0-9]+}}, [[R]], {{s[0-9]+}}, eq
}

define float @fueq(float %a, float %b) #0 {
  %t = fcmp ueq float %a, 0.0
  %v = select i1 %t, float 0.0, float %b
  ret float %v
; CHECK-LABEL: fueq
; CHECK: fcmp [[R:s[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{s[0-9]+}}, [[R]], {{s[0-9]+}}, eq
; CHECK-NEXT: fcsel {{s[0-9]+}}, [[R]], {{s[0-9]+}}, vs
}

define float @fone(float %a, float %b) #0 {
  %t = fcmp one float %a, 0.0
  %v = select i1 %t, float %b, float 0.0
  ret float %v
; CHECK-LABEL: fone
; CHECK: fcmp [[R:s[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{s[0-9]+}}, {{s[0-9]+}}, [[R]], mi
; CHECK-NEXT: fcsel {{s[0-9]+}}, {{s[0-9]+}}, [[R]], gt
}

define float @fune(float %a, float %b) #0 {
  %t = fcmp une float %a, 0.0
  %v = select i1 %t, float %b, float 0.0
  ret float %v
; CHECK-LABEL: fune
; CHECK: fcmp [[R:s[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{s[0-9]+}}, {{s[0-9]+}}, [[R]], ne
}

define double @doeq(double %a, double %b) #0 {
  %t = fcmp oeq double %a, 0.0
  %v = select i1 %t, double 0.0, double %b
  ret double %v
; CHECK-LABEL: doeq
; CHECK: fcmp [[R:d[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{d[0-9]+}}, [[R]], {{d[0-9]+}}, eq
}

define double @dueq(double %a, double %b) #0 {
  %t = fcmp ueq double %a, 0.0
  %v = select i1 %t, double 0.0, double %b
  ret double %v
; CHECK-LABEL: dueq
; CHECK: fcmp [[R:d[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{d[0-9]+}}, [[R]], {{d[0-9]+}}, eq
; CHECK-NEXT: fcsel {{d[0-9]+}}, [[R]], {{d[0-9]+}}, vs
}

define double @done(double %a, double %b) #0 {
  %t = fcmp one double %a, 0.0
  %v = select i1 %t, double %b, double 0.0
  ret double %v
; CHECK-LABEL: done
; CHECK: fcmp [[R:d[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{d[0-9]+}}, {{d[0-9]+}}, [[R]], mi
; CHECK-NEXT: fcsel {{d[0-9]+}}, {{d[0-9]+}}, [[R]], gt
}

define double @dune(double %a, double %b) #0 {
  %t = fcmp une double %a, 0.0
  %v = select i1 %t, double %b, double 0.0
  ret double %v
; CHECK-LABEL: dune
; CHECK: fcmp [[R:d[0-9]+]], #0.0
; CHECK-NEXT: fcsel {{d[0-9]+}}, {{d[0-9]+}}, [[R]], ne
}

attributes #0 = { nounwind "unsafe-fp-math"="true" }

