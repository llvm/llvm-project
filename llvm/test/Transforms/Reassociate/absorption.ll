; RUN: opt -S -reassociate < %s | FileCheck %s

; Check that if constants combine to an absorbing value then the expression is
; evaluated as the absorbing value.
define i8 @foo(i8 %x) {
  %tmp1 = or i8 %x, 127
  %tmp2 = or i8 %tmp1, 128
  ret i8 %tmp2
; CHECK-LABEL: @foo(
; CHECK: ret i8 -1
}
