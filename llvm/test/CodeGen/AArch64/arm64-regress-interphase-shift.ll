; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s

; This is mostly a "don't assert" test. The type of the RHS of a shift depended
; on the phase of legalization, which led to the creation of an unexpected and
; unselectable "rotr" node: (i32 (rotr i32, i64)).

; FIXME: This test is xfailed because it relies on an optimization that has
; been reverted (see PR17975).
; XFAIL: *

define void @foo(i64* nocapture %d) {
; CHECK-LABEL: foo:
; CHECK: rorv
  %tmp = load i64, i64* undef, align 8
  %sub397 = sub i64 0, %tmp
  %and398 = and i64 %sub397, 4294967295
  %shr404 = lshr i64 %and398, 0
  %or405 = or i64 0, %shr404
  %xor406 = xor i64 %or405, 0
  %xor417 = xor i64 0, %xor406
  %xor428 = xor i64 0, %xor417
  %sub430 = sub i64 %xor417, 0
  %and431 = and i64 %sub430, 4294967295
  %and432 = and i64 %xor428, 31
  %sub433 = sub i64 32, %and432
  %shl434 = shl i64 %and431, %sub433
  %shr437 = lshr i64 %and431, %and432
  %or438 = or i64 %shl434, %shr437
  %xor439 = xor i64 %or438, %xor428
  %sub441 = sub i64 %xor439, 0
  store i64 %sub441, i64* %d, align 8
  ret void
}
