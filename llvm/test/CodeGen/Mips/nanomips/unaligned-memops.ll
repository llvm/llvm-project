; RUN: llc -march=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define void @g2(i32* %a, i32* %b) {
; CHECK: ualw $a0, 0($a0)
; CHECK: UALW_NM
; CHECK: uasw $a0, 0($a1)
; CHECK: UASW_NM
  %1 = load i32, i32* %a, align 1
  store i32 %1, i32* %b, align 1
  ret void
}
