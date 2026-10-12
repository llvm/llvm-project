; RUN: opt -S -passes='sink<avoid-postdominators>' < %s | FileCheck %s --check-prefixes=CHECK,SCAN
; RUN: opt -S -passes='sink<avoid-postdominators>' -sink-max-uses-to-scan=2 < %s | FileCheck %s --check-prefixes=CHECK,LIMIT

; %x has more uses than the limit, so it is assumed to be live into %merge and
; only %y counts as extended, which does not outweigh freeing %m.
define float @operand_with_many_uses(float %x, float %y, i1 %c, ptr %p) {
; CHECK-LABEL: @operand_with_many_uses(
; CHECK-NEXT:  entry:
; SCAN-NEXT:     %m = fmul float %x, %y
; CHECK-NEXT:    br i1 %c, label %then, label %merge
; CHECK:       merge:
; LIMIT-NEXT:    %m = fmul float %x, %y
; CHECK-NEXT:    ret float %m
;
entry:
  %m = fmul float %x, %y
  br i1 %c, label %then, label %merge

then:
  store float %x, ptr %p
  store float %x, ptr %p
  br label %merge

merge:
  ret float %m
}
