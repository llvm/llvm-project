; RUN: opt -S -mtriple=x86_64-- -expand-ir-insts -expand-div-rem-bits 32 < %s | FileCheck %s
; RUN: opt -S -mtriple=x86_64-- -passes='require<libcall-lowering-info>,expand-ir-insts' -expand-div-rem-bits 32 < %s | FileCheck %s

; With 16-bit limbs, a divisor above 2^16 - 1 keeps the bit-serial expansion.

define i64 @udiv_i64_by_2pow16_plus_1(i64 %x) {
; CHECK-LABEL: @udiv_i64_by_2pow16_plus_1(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %r = udiv i64 %x, 65537
  ret i64 %r
}
