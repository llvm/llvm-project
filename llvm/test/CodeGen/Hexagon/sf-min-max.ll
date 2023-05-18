; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: sf_min_olt:
; CHECK: sfmin
define float @sf_min_olt(float %x, float %y) #0 {
  %t = fcmp olt float %x, %y
  %u = select i1 %t, float %x, float %y
  ret float %u
}

; CHECK-LABEL: sf_min_ole:
; CHECK: sfmin
define float @sf_min_ole(float %x, float %y) #0 {
  %t = fcmp ole float %x, %y
  %u = select i1 %t, float %x, float %y
  ret float %u
}

; CHECK-LABEL: sf_max_ogt:
; CHECK: sfmax
define float @sf_max_ogt(float %x, float %y) #0 {
  %t = fcmp ogt float %x, %y
  %u = select i1 %t, float %x, float %y
  ret float %u
}

; CHECK-LABEL: sf_max_oge:
; CHECK: sfmax
define float @sf_max_oge(float %x, float %y) #0 {
  %t = fcmp oge float %x, %y
  %u = select i1 %t, float %x, float %y
  ret float %u
}

; CHECK-LABEL: sf_max_olt:
; CHECK: sfmax
define float @sf_max_olt(float %x, float %y) #0 {
  %t = fcmp olt float %x, %y
  %u = select i1 %t, float %y, float %x
  ret float %u
}

; CHECK-LABEL: sf_max_ole:
; CHECK: sfmax
define float @sf_max_ole(float %x, float %y) #0 {
  %t = fcmp ole float %x, %y
  %u = select i1 %t, float %y, float %x
  ret float %u
}

; CHECK-LABEL: sf_min_ogt:
; CHECK: sfmin
define float @sf_min_ogt(float %x, float %y) #0 {
  %t = fcmp ogt float %x, %y
  %u = select i1 %t, float %y, float %x
  ret float %u
}

; CHECK-LABEL: sf_min_oge:
; CHECK: sfmin
define float @sf_min_oge(float %x, float %y) #0 {
  %t = fcmp oge float %x, %y
  %u = select i1 %t, float %y, float %x
  ret float %u
}

attributes #0 = { nounwind "target-cpu"="hexagonv5" }
