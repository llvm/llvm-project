; RUN: llc < %s -mtriple=mipsel -mcpu=mips1 -relocation-model=static \
; RUN:   | FileCheck %s -check-prefixes=ALL,DELAY
; RUN: llc < %s -mtriple=mipsel -mcpu=mips2 -relocation-model=static \
; RUN:   | FileCheck %s -check-prefixes=ALL,DELAY
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -relocation-model=static \
; RUN:   | FileCheck %s -check-prefixes=ALL,INTERLOCK

; On MIPS-I to MIPS-III a transfer out of the FPU is not interlocked: the
; instruction executed after an "mfc1" must not read its destination. Filling
; a branch delay slot with the mfc1 moves that shadow onto the branch target
; and the fall-through, so the delay slot filler must not do it there.
; From MIPS32 on the hardware interlocks and the slot may be filled.

define i32 @xfer(float %a, float %b, i32 %sel) nounwind {
; ALL-LABEL: xfer:
entry:
  %x = bitcast float %a to i32
  %y = bitcast float %b to i32
  %cmp = icmp ne i32 %sel, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %add = add i32 %x, %y
  ret i32 %add

if.else:
  %sub = sub i32 %x, %y
  ret i32 %sub

; The instruction in the branch delay slot must not be an mfc1.
; DELAY:          {{b(ne|eq)z}} ${{[0-9]+}},
; DELAY-NEXT:     nop
; INTERLOCK:      {{b(ne|eq)z}} ${{[0-9]+}},
; INTERLOCK-NEXT: mfc1
}
