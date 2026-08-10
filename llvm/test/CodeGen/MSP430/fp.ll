; RUN: llc -O0 -frame-pointer=all < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

define void @fp() nounwind {
entry:
; CHECK-LABEL: fp:
; CHECK: push r4
; CHECK: mov r1, r4
; CHECK: sub #2, r1
  %i = alloca i16, align 2
; CHECK: clr -2(r4)
  store i16 0, ptr %i, align 2
; CHECK: pop r4
  ret void
}

; Due to FPB not being marked as reserved, the register allocator used to select
; r4 as the register for the "r" constraint below. This test verifies that this
; does not happen anymore. Note that the only reason an ISR is used here is that
; the register allocator selects r4 first instead of fifth in a normal function.
define msp430_intrcc void @fpb_alloced() #0 {
; CHECK-LABEL: fpb_alloced:
; CHECK-NOT: mov.b #0, r4
; CHECK: nop
  call void asm sideeffect "nop", "r"(i8 0)
  ret void
}

attributes #0 = { noinline nounwind optnone "interrupt"="2" }
