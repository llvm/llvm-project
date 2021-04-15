; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define i32 @foo0() nounwind readnone {
; CHECK-LABEL: foo0
entry:
; CHECK: li $a0, 12345
; CHECK: Li_NM
  ret i32 12345
}

define i32 @foo1() nounwind readnone {
; CHECK-LABEL: foo1
entry:
; CHECK: li $a0, -2147483648
; CHECK: Li_NM
  ret i32 -2147483648
}

define i32 @foo2() nounwind readnone {
; CHECK-LABEL: foo2
entry:
; CHECK: li $a0, 2147483647
; CHECK: Li_NM
  ret i32 2147483647
}
