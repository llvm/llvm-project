; RUN: llc < %s -march=mips -mcpu=mips2 | FileCheck %s -check-prefix=ALL

; This test triggered a bug in the vector splitting where the type legalizer
; attempted to extract the element with by storing the vector, then reading
; an element back. However, the address calculation was:
;   Base + Index * (EltSizeInBits / 8)
; and EltSizeInBits was 1. This caused the index to be forgotten.
define i1 @via_stack_bug(i8 signext %idx) {
  %1 = extractelement <2 x i1> <i1 false, i1 true>, i8 %idx
  ret i1 %1
}

; ALL-LABEL: via_stack_bug:
; ALL-DAG:       addiu  [[ONE:\$[0-9]+]], $zero, 1
; ALL-DAG:       sb     [[ONE]], 7($sp)
; ALL-DAG:       sb     $zero, 6($sp)
; ALL-DAG:       andi   [[MASKED_IDX:\$[0-9]+]], $4, 1
; ALL-DAG:       addiu  [[VPTR:\$[0-9]+]], $sp, 6
; ALL-DAG:       or   [[EPTR:\$[0-9]+]], [[MASKED_IDX]], [[VPTR]]
; ALL:           lbu    $2, 0([[EPTR]])
