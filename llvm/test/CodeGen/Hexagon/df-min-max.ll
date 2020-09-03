; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: df_min_olt:
; CHECK: dfmin
define double @df_min_olt(double %x, double %y) #0 {
  %t = fcmp olt double %x, %y
  %u = select i1 %t, double %x, double %y
  ret double %u
}

; CHECK-LABEL: df_min_ole:
; CHECK: dfmin
define double @df_min_ole(double %x, double %y) #0 {
  %t = fcmp ole double %x, %y
  %u = select i1 %t, double %x, double %y
  ret double %u
}

; CHECK-LABEL: df_max_ogt:
; CHECK: dfmax
define double @df_max_ogt(double %x, double %y) #0 {
  %t = fcmp ogt double %x, %y
  %u = select i1 %t, double %x, double %y
  ret double %u
}

; CHECK-LABEL: df_max_oge:
; CHECK: dfmax
define double @df_max_oge(double %x, double %y) #0 {
  %t = fcmp oge double %x, %y
  %u = select i1 %t, double %x, double %y
  ret double %u
}

; CHECK-LABEL: df_max_olt:
; CHECK: dfmax
define double @df_max_olt(double %x, double %y) #0 {
  %t = fcmp olt double %x, %y
  %u = select i1 %t, double %y, double %x
  ret double %u
}

; CHECK-LABEL: df_max_ole:
; CHECK: dfmax
define double @df_max_ole(double %x, double %y) #0 {
  %t = fcmp ole double %x, %y
  %u = select i1 %t, double %y, double %x
  ret double %u
}

; CHECK-LABEL: df_min_ogt:
; CHECK: dfmin
define double @df_min_ogt(double %x, double %y) #0 {
  %t = fcmp ogt double %x, %y
  %u = select i1 %t, double %y, double %x
  ret double %u
}

; CHECK-LABEL: df_min_oge:
; CHECK: dfmin
define double @df_min_oge(double %x, double %y) #0 {
  %t = fcmp oge double %x, %y
  %u = select i1 %t, double %y, double %x
  ret double %u
}

attributes #0 = { nounwind "target-cpu"="hexagonv67" }
