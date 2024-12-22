; RUN: llc < %s -march=mips64 -mcpu=mips3 | FileCheck %s

; Currently, the following IR assembly generates a KILL instruction between
; the bitwise-and instruction and the return instruction. We verify that the
; delay slot filler ignores such KILL instructions by filling the slot of the
; return instruction properly.
define signext i32 @f1(i32 signext %a, i32 signext %b) {
entry:
  ; CHECK:          jr      $ra
  ; CHECK-NEXT:     and     $2, $4, $5

  %r = and i32 %a, %b
  ret i32 %r
}
