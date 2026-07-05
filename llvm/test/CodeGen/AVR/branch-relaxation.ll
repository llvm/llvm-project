; RUN: llc < %s -mtriple=avr | FileCheck %s
; RUN: llc < %s -mtriple=avr -mcpu=avr5 | FileCheck -check-prefix=AVR5 %s

; CHECK-LABEL: relax_breq
; CHECK: cpi     r{{[0-9]+}}, 0
; CHECK: brne    .LBB0_1
; CHECK: rjmp    .LBB0_2
; CHECK: .LBB0_1:
; CHECK: nop
; CHECK: .LBB0_2:

; AVR5-LABEL: relax_breq
; AVR5:         andi    r24, 1
; AVR5:         cpi     r24, 0
; AVR5:         brne    .LBB0_1
; AVR5:         rjmp    .LBB0_2
; AVR5:       .LBB0_1:
; AVR5:         nop
; AVR5:       .LBB0_2:

define i8 @relax_breq(i1 %a) {
entry-block:
  br i1 %a, label %hello, label %finished

hello:
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  br label %finished
finished:
  ret i8 3
}

; CHECK-LABEL: no_relax_breq
; CHECK: cpi     r{{[0-9]+}}, 0
; CHECK: breq    [[END_BB:.LBB[0-9]+_[0-9]+]]
; CHECK: nop
; CHECK: [[END_BB]]

; AVR5-LABEL: no_relax_breq
; AVR5:         cpi     r{{[0-9]+}}, 0
; AVR5:         breq    [[END_BB:.LBB[0-9]+_[0-9]+]]
; AVR5:         nop
; AVR5:       [[END_BB]]

define i8 @no_relax_breq(i1 %a) {
entry-block:
  br i1 %a, label %hello, label %finished

hello:
  ; There are not enough NOPs to require relaxation.
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  call void asm sideeffect "nop", ""()
  br label %finished
finished:
  ret i8 3
}

