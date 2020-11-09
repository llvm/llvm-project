; RUN: llc -mtriple=arm-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - -O3 \
; RUN:  -asm-verbose=0 | FileCheck %s

; This tests exerts the folding of `VT = (and (sign_extend NarrowVT to
; VT) #bitmask)` into `VT = (zero_extend NarrowVT to VT)` when
; #bitmask value is the mask made by all ones that selects the value
; of type NarrowVT inside the value of type VT. The folding is
; implemented in `DAGCombiner::visitAND`.

; With this the folding, the `and` of the "signed extended load" of
; `%b` in `f_i16_i32` is rendered as a zero extended load.

; CHECK-LABEL: f_i16_i32:
; CHECK-NEXT: .fnstart
; CHECK-NEXT: ldrh    r1, [r1]
; CHECK-NEXT: ldrsh   r0, [r0]
; CHECK-NEXT: smulbb  r0, r0, r1
; CHECK-NEXT: mul     r0, r0, r1
; CHECK-NEXT: bx      lr
define i32 @f_i16_i32(i16* %a, i16* %b) {
  %1 = load i16, i16* %a, align 2
  %sext.1 = sext i16 %1 to i32
  %2 = load i16, i16* %b, align 2
  %sext.2 = sext i16 %2 to i32
  %masked = and i32 %sext.2, 65535
  %mul = mul nsw i32 %sext.2, %sext.1
  %count.next = mul i32 %mul, %masked
  ret i32 %count.next
}
