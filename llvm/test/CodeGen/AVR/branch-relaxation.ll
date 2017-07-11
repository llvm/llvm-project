; RUN: llc < %s -march=avr | FileCheck %s

; CHECKC-LABEL: relax_breq
; CHECK: cpi     r{{[0-9]+}}, 0
; CHECK: brne    LBB0_1
; CHECK: rjmp    LBB0_2
; LBB0_1:

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

; CHECKC-LABEL: no_relax_breq
; CHECK: cpi     r{{[0-9]+}}, 0
; CHECK: breq    [[END_BB:LBB[0-9]+_[0-9]+]]
; CHECK: nop
; ...
; LBB0_1:
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

