; RUN: opt < %s -cost-model -analyze -mtriple=systemz-unknown -mcpu=z13 | FileCheck %s
;
; Test that loads into operations that can fold one memory operand get zero
; cost. In the case that both operands are loaded, one load should get a cost
; value.

define void @add() {
  %li32 = load i32, i32* undef
  add i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  add i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  add i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  add i64 %li64_0, %li64_1

  ret void;

; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = add i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = add i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = add i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = add i64 %li64_0, %li64_1
}

define void @sub() {
  %li32 = load i32, i32* undef
  sub i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  sub i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  sub i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  sub i64 %li64_0, %li64_1

  ret void;

; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = sub i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = sub i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = sub i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = sub i64 %li64_0, %li64_1
}

define void @mul() {
  %li32 = load i32, i32* undef
  mul i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  mul i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  mul i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  mul i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = mul i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = mul i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = mul i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = mul i64 %li64_0, %li64_1
}

define void @sdiv() {
  %li32 = load i32, i32* undef
  sdiv i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  sdiv i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  sdiv i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  sdiv i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %1 = sdiv i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %2 = sdiv i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = sdiv i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = sdiv i64 %li64_0, %li64_1
}

define void @udiv() {
  %li32 = load i32, i32* undef
  udiv i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  udiv i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  udiv i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  udiv i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %1 = udiv i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %2 = udiv i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %3 = udiv i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 2 for instruction:   %4 = udiv i64 %li64_0, %li64_1
}

define void @and() {
  %li32 = load i32, i32* undef
  and i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  and i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  and i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  and i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = and i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = and i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = and i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = and i64 %li64_0, %li64_1
}

define void @or() {
  %li32 = load i32, i32* undef
  or i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  or i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  or i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  or i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = or i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = or i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = or i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = or i64 %li64_0, %li64_1
}

define void @xor() {
  %li32 = load i32, i32* undef
  xor i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  xor i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  xor i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  xor i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = xor i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = xor i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = xor i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = xor i64 %li64_0, %li64_1
}

define void @icmp() {
  %li32 = load i32, i32* undef
  icmp eq i32 %li32, undef

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  icmp eq i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  icmp eq i64 %li64, undef

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  icmp eq i64 %li64_0, %li64_1

  ret void;
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = icmp eq i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = icmp eq i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = icmp eq i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = icmp eq i64 %li64_0, %li64_1
}
