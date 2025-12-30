; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define float @negateF(float %x) {
; CHECK-LABEL: define float @negateF(
; CHECK-SAME: float [[X:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[Y:%.*]] = fsub float -0.000000e+00, [[X]]
; CHECK-NEXT:    ret float [[Y]]
entry:  
  %y = fneg float %x
  ret float %y
}

define double @negateD(double %x) {
; CHECK-LABEL: define double @negateD(
; CHECK-SAME: double [[X:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[Y:%.*]] = fsub double -0.000000e+00, [[X]]
; CHECK-NEXT:    ret double [[Y]]
entry:  
  %y = fneg double %x
  ret double %y
}
