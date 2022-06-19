; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; ScalarEvolution shouldn't attempt to interpret expressions which have
; undefined results.

define void @foo(i64 %x) {

  %a = udiv i64 %x, 0
; CHECK: -->  (%x /u 0)

  %B = shl i64 %x, 64
; CHECK: -->  %B

  %b = ashr i64 %B, 64
; CHECK: -->  %b

  %c = lshr i64 %x, 64
; CHECK: -->  %c

  %d = shl i64 %x, 64
; CHECK: -->  %d

  %E = shl i64 %x, -1
; CHECK: -->  %E

  %e = ashr i64 %E, -1
; CHECK: -->  %e

  %f = lshr i64 %x, -1
; CHECK: -->  %f

  %g = shl i64 %x, -1
; CHECK: -->  %g

  %h = bitcast i64 undef to i64
; CHECK: -->  undef

  ret void
}
