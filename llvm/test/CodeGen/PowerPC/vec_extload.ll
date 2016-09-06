; RUN: llc -verify-machineinstrs -mcpu=pwr6 -mattr=+altivec -code-model=small < %s | FileCheck %s

; Check vector extend load expansion with altivec enabled.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Altivec does not provides an sext instruction, so it expands
; a set of vector stores (stvx), bytes load/sign expand/store
; (lbz/stb), and a final vector load (lvx) to load the result
; extended vector.
define <16 x i8> @v16si8_sext_in_reg(<16 x i8> %a) {
  %b = trunc <16 x i8> %a to <16 x i4>
  %c = sext <16 x i4> %b to <16 x i8>
  ret <16 x i8> %c
}
; CHECK-LABEL: v16si8_sext_in_reg:
; CHECK: vslb
; CHECK: vsrab
; CHECK: blr 

; The zero extend uses a more clever logic: a vector splat
; and a logic and to set higher bits to 0.
define <16 x i8> @v16si8_zext_in_reg(<16 x i8> %a) {
  %b = trunc <16 x i8> %a to <16 x i4>
  %c = zext <16 x i4> %b to <16 x i8>
  ret <16 x i8> %c
}
; CHECK-LABEL:      v16si8_zext_in_reg:
; CHECK:      vspltisb [[VMASK:[0-9]+]], 15
; CHECK-NEXT: vand 2, 2, [[VMASK]]

; Same as v16si8_sext_in_reg, expands to load/store halfwords (lhz/sth).
define <8 x i16> @v8si16_sext_in_reg(<8 x i16> %a) {
  %b = trunc <8 x i16> %a to <8 x i8>
  %c = sext <8 x i8> %b to <8 x i16>
  ret <8 x i16> %c
}
; CHECK-LABEL: v8si16_sext_in_reg:
; CHECK: vslh
; CHECK: vsrah
; CHECK: blr 

; Same as v8si16_sext_in_reg, but instead of creating the mask
; with a splat, loads it from memory.
define <8 x i16> @v8si16_zext_in_reg(<8 x i16> %a) {
  %b = trunc <8 x i16> %a to <8 x i8>
  %c = zext <8 x i8> %b to <8 x i16>
  ret <8 x i16> %c
}
; CHECK-LABEL:      v8si16_zext_in_reg:
; CHECK:      ld [[RMASKTOC:[0-9]+]], .LC{{[0-9]+}}@toc(2)
; CHECK-NEXT: lvx [[VMASK:[0-9]+]], {{[0-9]+}}, [[RMASKTOC]]
; CHECK-NEXT: vand 2, 2, [[VMASK]]

; Same as v16si8_sext_in_reg, expands to load halfword (lha) and
; store words (stw).
define <4 x i32> @v4si32_sext_in_reg(<4 x i32> %a) {
  %b = trunc <4 x i32> %a to <4 x i16>
  %c = sext <4 x i16> %b to <4 x i32>
  ret <4 x i32> %c
}
; CHECK-LABEL: v4si32_sext_in_reg:
; CHECK: vslw
; CHECK: vsraw
; CHECK: blr 

; Same as v8si16_sext_in_reg.
define <4 x i32> @v4si32_zext_in_reg(<4 x i32> %a) {
  %b = trunc <4 x i32> %a to <4 x i16>
  %c = zext <4 x i16> %b to <4 x i32>
  ret <4 x i32> %c
}
; CHECK-LABEL:      v4si32_zext_in_reg:
; CHECK:      vspltisw [[VMASK:[0-9]+]], -16
; CHECK-NEXT: vsrw [[VMASK]], [[VMASK]], [[VMASK]]
; CHECK-NEXT: vand 2, 2, [[VMASK]]
