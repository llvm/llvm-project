; RUN: llvm-as < %s | llvm-dis | FileCheck %s

define float @nan(float %a) {
; CHECK-LABEL: define float @nan(
; CHECK-SAME: float [[A:%.*]]) {
; CHECK-NEXT:    [[T:%.*]] = fadd float [[A]], 0x7FF8000000000000
; CHECK-NEXT:    [[T1:%.*]] = fadd float [[T]], 0x7FFA000000000000
; CHECK-NEXT:    [[R:%.*]] = fadd float [[T1]], 0x7FF4000000000000
; CHECK-NEXT:    ret float [[T1]]
;
  %t = fadd float %a, nan
  %t1 = fadd float %t, qnan(u0x2000000000000)
  %r = fadd float %t1, snan(u0x4000000000000)
  ret float %t1
}

define float @inf(float %a) {
; CHECK-LABEL: define float @inf(
; CHECK-SAME: float [[A:%.*]]) {
; CHECK-NEXT:    [[T:%.*]] = fadd float [[A]], 0x7FF0000000000000
; CHECK-NEXT:    [[R:%.*]] = fadd float [[T]], 0xFFF0000000000000
; CHECK-NEXT:    ret float [[R]]
;
  %t = fadd float %a, pinf
  %r = fadd float %t, ninf
  ret float %r
}
