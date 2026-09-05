; RUN: opt -S -mtriple=x86_64-- -expand-ir-insts -expand-div-rem-bits 128 < %s | FileCheck %s
; RUN: opt -S -mtriple=x86_64-- -passes='require<libcall-lowering-info>,expand-ir-insts' -expand-div-rem-bits 128 < %s | FileCheck %s

; Operations whose divisor is not known to fit in a 32-bit limb, and signed
; operations, keep the generic bit-serial expansion.

define i256 @udiv_i256_by_2pow32_plus_1(i256 %x) {
; CHECK-LABEL: @udiv_i256_by_2pow32_plus_1(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %r = udiv i256 %x, 4294967297
  ret i256 %r
}

define i256 @urem_i256_by_variable(i256 %x, i256 %d) {
; CHECK-LABEL: @urem_i256_by_variable(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %r = urem i256 %x, %d
  ret i256 %r
}

define i256 @udiv_i256_by_zext_i33(i256 %x, i33 %d) {
; CHECK-LABEL: @udiv_i256_by_zext_i33(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %d.wide = zext i33 %d to i256
  %r = udiv i256 %x, %d.wide
  ret i256 %r
}

define i256 @sdiv_i256_by_100(i256 %x) {
; CHECK-LABEL: @sdiv_i256_by_100(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %r = sdiv i256 %x, 100
  ret i256 %r
}

define i256 @srem_i256_by_100(i256 %x) {
; CHECK-LABEL: @srem_i256_by_100(
; CHECK-NOT:   udivrem-limb-loop
; CHECK:       udiv-do-while:
; CHECK-NOT:   udivrem-limb-loop
  %r = srem i256 %x, 100
  ret i256 %r
}
