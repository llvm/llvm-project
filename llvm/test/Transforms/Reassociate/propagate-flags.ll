; RUN: opt < %s -reassociate -S | FileCheck %s

define double @func(double %a, double %b) {
; CHECK-LABEL: @func(
; CHECK-NEXT:    [[TMP1:%.*]] = fmul fast double %b, %a
; CHECK-NEXT:    [[TMP2:%.*]] = fmul fast double [[TMP1]], [[TMP1]]
; CHECK-NEXT:    ret double [[TMP2]]
;
  %mul1 = fmul fast double %a, %a
  %mul2 = fmul fast double %b, %b
  %mul3 = fmul fast double %mul1, %mul2
  ret double %mul3
}
