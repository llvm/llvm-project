; RUN: llc %s -march=mipsel -mcpu=mips32r2 -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -o - | FileCheck %s

@x = common global i32 0, align 4

define i32 @AtomicLoadAdd32(i32 %incr) nounwind {
entry:
  %0 = atomicrmw add i32* @x, i32 %incr monotonic
  ret i32 %0

; CHECK-LABEL:   AtomicLoadAdd32:
; CHECK:   lw      $[[R0:[0-9]+]], %got(x)
; CHECK:   $[[BB0:[A-Z_0-9]+]]:
; CHECK:   ll      $[[R1:[0-9]+]], 0($[[R0]])
; CHECK:   addu    $[[R2:[0-9]+]], $[[R1]], $4
; CHECK:   sc      $[[R2]], 0($[[R0]])
; CHECK:   beqzc   $[[R2]], $[[BB0]]
}
