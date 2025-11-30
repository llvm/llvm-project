; RUN: opt -passes=instcombine -S %s | FileCheck %s

; ================================================================
; Test: Simplify zext(sub(0, trunc(x))) -> and(sub(0, x), (bitwidth-1))
; Purpose: Check that InstCombine detects and simplifies the pattern
;          seen in rotate idioms, enabling backend rotate lowering.
; ================================================================

; === Scalar Case (i64) =========================================
define i64 @neg_trunc_zext(i64 %a) {
; CHECK-LABEL: @neg_trunc_zext(
; CHECK-NEXT: %[[NEG:[0-9]+]] = sub i64 0, %a
; CHECK-NEXT: %[[MASKED:[0-9A-Za-z_]+]] = and i64 %[[NEG]], 63
; CHECK-NEXT: ret i64 %[[MASKED]]
  %t = trunc i64 %a to i6
  %n = sub i6 0, %t
  %z = zext i6 %n to i64
  ret i64 %z
}

; === Vector Case 1: <2 x i64> ==================================
define <2 x i64> @foo(<2 x i64> %x, <2 x i64> %n) {
; CHECK-LABEL: @foo(
; CHECK: %[[NEG:[0-9A-Za-z_]+]] = sub <2 x i64> zeroinitializer, %n
; CHECK: %[[MASK:[0-9A-Za-z_]+]] = and <2 x i64> %[[NEG]], splat (i64 63)
; CHECK: ret <2 x i64> %[[MASK]]
  %t = trunc <2 x i64> %n to <2 x i6>
  %neg = sub <2 x i6> zeroinitializer, %t
  %z = zext <2 x i6> %neg to <2 x i64>
  ret <2 x i64> %z
}

; === Vector Case 2: <4 x i64> ==================================
define <4 x i64> @bar(<4 x i64> %x, <4 x i64> %n) {
; CHECK-LABEL: @bar(
; CHECK: %[[NEG:[0-9A-Za-z_]+]] = sub <4 x i64> zeroinitializer, %n
; CHECK: %[[MASK:[0-9A-Za-z_]+]] = and <4 x i64> %[[NEG]], splat (i64 63)
; CHECK: ret <4 x i64> %[[MASK]]
  %t = trunc <4 x i64> %n to <4 x i6>
  %neg = sub <4 x i6> zeroinitializer, %t
  %z = zext <4 x i6> %neg to <4 x i64>
  ret <4 x i64> %z
}

; === Vector Case 3: <8 x i64> ==================================
define <8 x i64> @baz(<8 x i64> %x, <8 x i64> %n) {
; CHECK-LABEL: @baz(
; CHECK: %[[NEG:[0-9A-Za-z_]+]] = sub <8 x i64> zeroinitializer, %n
; CHECK: %[[MASK:[0-9A-Za-z_]+]] = and <8 x i64> %[[NEG]], splat (i64 63)
; CHECK: ret <8 x i64> %[[MASK]]
  %t = trunc <8 x i64> %n to <8 x i6>
  %neg = sub <8 x i6> zeroinitializer, %t
  %z = zext <8 x i6> %neg to <8 x i64>
  ret <8 x i64> %z
}
