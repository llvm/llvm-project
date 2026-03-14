; RUN: llc -mtriple=thumb-none-macho -mcpu=arm7tdmi %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb-none-macho -mcpu=arm7tdmi %s -filetype=obj -o /dev/null

declare void @callee()

define void @test_call() {
  ; BX can only take a register before v5t came along, so we must materialise
  ; the address properly.
; CHECK-LABEL: test_call:
; CHECK: ldr r[[CALLEE_STUB:[0-9]+]], [[LITPOOL:LCPI[0-9]+_[0-9]+]]
; CHECK: [[PC_LABEL:LPC[0-9]+_[0-9]+]]:
; CHECK-NEXT: add r[[CALLEE_STUB]], pc
; CHECK: ldr [[CALLEE:r[0-9]+]], [r[[CALLEE_STUB]]]
; CHECK-NOT: mov lr, pc
; CHECK: bl [[INDIRECT_PAD:Ltmp[0-9]+]]

; CHECK: [[LITPOOL]]:
; CHECK-NEXT: .long L_callee$non_lazy_ptr-([[PC_LABEL]]+4)

; CHECK: [[INDIRECT_PAD]]:
; CHECK: bx [[CALLEE]]

  call void @callee()
  ret void
}
