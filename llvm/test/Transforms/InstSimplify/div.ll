; RUN: opt < %s -instsimplify -S | FileCheck %s

declare i32 @external()

define i32 @div1() {
; CHECK-LABEL: @div1(
; CHECK:         [[CALL:%.*]] = call i32 @external(), !range !0
; CHECK-NEXT:    ret i32 0
;
  %call = call i32 @external(), !range !0
  %urem = udiv i32 %call, 3
  ret i32 %urem
}

!0 = !{i32 0, i32 3}
