; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z13 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z13
; RUN: opt < %s -passes="print<cost-model>" 2>&1 -disable-output -mtriple=systemz-unknown -mcpu=z14 \
; RUN:  | FileCheck %s -check-prefixes=CHECK,Z14
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  add i32 %tr, undef

  ; Sign-extended loads
  %li16_0 = load i16, i16* undef
  %sext_0 = sext i16 %li16_0 to i32
  add i32 %sext_0, undef

  %li16_1 = load i16, i16* undef
  %sext_1 = sext i16 %li16_1 to i64
  add i64 %sext_1, undef

  %li32_2 = load i32, i32* undef
  %sext_2 = sext i32 %li32_2 to i64
  add i64 %sext_2, undef

  ; Zero-extended loads
  %li32_3 = load i32, i32* undef
  %zext_0 = zext i32 %li32_3 to i64
  add i64 %zext_0, undef

  ; Loads with multiple uses are *not* folded
  %li16_3 = load i16, i16* undef
  %sext_3 = sext i16 %li16_3 to i32
  %sext_4 = sext i16 %li16_3 to i32
  add i32 %sext_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = add i32 %tr, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li16_0 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i16 %li16_0 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = add i32 %sext_0, undef
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %li16_1 = load i16, i16* undef
; Z14:   Cost Model: Found an estimated cost of 0 for instruction:   %li16_1 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i16 %li16_1 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = add i64 %sext_1, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = add i64 %sext_2, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %zext_0 = zext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = add i64 %zext_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_3 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_3 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_4 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = add i32 %sext_3, undef
}

define void @add_i16_mem16(i16 %Arg, i16* %Src1, i16* %Src2, i16* %Dst, i32* %Src32) {
  %L1 = load i16, i16* %Src1
  %S0 = add i16 %L1, %Arg
  store volatile i16 %S0, i16* %Dst

  %L2 = load i16, i16* %Src1
  %L3 = load i16, i16* %Src2
  %S1 = add i16 %L2, %L3
  store volatile i16 %S1, i16* %Dst

  ; Truncated load
  %L32 = load i32, i32* %Src32
  %tr = trunc i32 %L32 to i16
  %S2 = add i16 %tr, %Arg
  store volatile i16 %S2, i16* %Dst

  ret void
; CHECK: function 'add_i16_mem16'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L1 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %S0 = add i16 %L1, %Arg
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %S0, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L2 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %L3 = load i16, i16* %Src2
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %S1 = add i16 %L2, %L3
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %S1, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L32 = load i32, i32* %Src32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i32 %L32 to i16
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %S2 = add i16 %tr, %Arg
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %S2, i16* %Dst
}

define void @sub_lhs_mem() {
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  sub i32 %tr, undef

  ; Sign-extended loads
  %li16_0 = load i16, i16* undef
  %sext_0 = sext i16 %li16_0 to i32
  sub i32 %sext_0, undef

  %li16_1 = load i16, i16* undef
  %sext_1 = sext i16 %li16_1 to i64
  sub i64 %sext_1, undef

  %li32_2 = load i32, i32* undef
  %sext_2 = sext i32 %li32_2 to i64
  sub i64 %sext_2, undef

  ; Zero-extended loads
  %li32_3 = load i32, i32* undef
  %zext_0 = zext i32 %li32_3 to i64
  sub i64 %zext_0, undef

  ; Loads with multiple uses are *not* folded
  %li16_3 = load i16, i16* undef
  %sext_3 = sext i16 %li16_3 to i32
  %sext_4 = sext i16 %li16_3 to i32
  sub i32 %sext_3, undef

  ret void;

; A sub LHS loaded operand is *not* foldable.
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = sub i32 %li32, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = sub i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = sub i64 %li64, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = sub i64 %li64_0, %li64_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = sub i32 %tr, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_0 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i16 %li16_0 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = sub i32 %sext_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_1 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i16 %li16_1 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = sub i64 %sext_1, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = sub i64 %sext_2, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %zext_0 = zext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = sub i64 %zext_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_3 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_3 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_4 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = sub i32 %sext_3, undef
}

define void @sub_rhs_mem() {
  %li32 = load i32, i32* undef
  sub i32 undef, %li32

  %li64 = load i64, i64* undef
  sub i64 undef, %li64

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  sub i32 undef, %tr

  ; Sign-extended loads
  %li16_0 = load i16, i16* undef
  %sext_0 = sext i16 %li16_0 to i32
  sub i32 undef, %sext_0

  %li16_1 = load i16, i16* undef
  %sext_1 = sext i16 %li16_1 to i64
  sub i64 undef, %sext_1

  %li32_2 = load i32, i32* undef
  %sext_2 = sext i32 %li32_2 to i64
  sub i64 undef, %sext_2

  ; Zero-extended loads
  %li32_3 = load i32, i32* undef
  %zext_0 = zext i32 %li32_3 to i64
  sub i64 undef, %zext_0

  ; Loads with multiple uses are *not* folded
  %li16_3 = load i16, i16* undef
  %sext_3 = sext i16 %li16_3 to i32
  %sext_4 = sext i16 %li16_3 to i32
  sub i32 undef, %sext_3

  ret void;

; A sub RHS loaded operand is foldable.
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %1 = sub i32 undef, %li32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %2 = sub i64 undef, %li64
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %3 = sub i32 undef, %tr
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li16_0 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i16 %li16_0 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %4 = sub i32 undef, %sext_0
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %li16_1 = load i16, i16* undef
; Z14:   Cost Model: Found an estimated cost of 0 for instruction:   %li16_1 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i16 %li16_1 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = sub i64 undef, %sext_1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = sub i64 undef, %sext_2
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %zext_0 = zext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = sub i64 undef, %zext_0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_3 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_3 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_4 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = sub i32 undef, %sext_3
}

define void @sub_i16_mem16(i16 %Arg, i16* %Src1, i16* %Src2, i16* %Dst, i32* %Src32) {
  %L1 = load i16, i16* %Src1
  %D0 = sub i16 %Arg, %L1
  store volatile i16 %D0, i16* %Dst

  %L2 = load i16, i16* %Src1
  %L3 = load i16, i16* %Src2
  %D1 = sub i16 %L2, %L3
  store volatile i16 %D1, i16* %Dst

  ; Truncated load
  %L32 = load i32, i32* %Src32
  %tr = trunc i32 %L32 to i16
  %D2 = sub i16 %Arg, %tr
  store volatile i16 %D2, i16* %Dst

  ret void
; CHECK: function 'sub_i16_mem16'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L1 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %D0 = sub i16 %Arg, %L1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %D0, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %L2 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L3 = load i16, i16* %Src2
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %D1 = sub i16 %L2, %L3
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %D1, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L32 = load i32, i32* %Src32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i32 %L32 to i16
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %D2 = sub i16 %Arg, %tr
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %D2, i16* %Dst
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  mul i32 %tr, undef

  ; Sign-extended loads
  %li16_0 = load i16, i16* undef
  %sext_0 = sext i16 %li16_0 to i32
  mul i32 %sext_0, undef

  %li16_1 = load i16, i16* undef
  %sext_1 = sext i16 %li16_1 to i64
  mul i64 %sext_1, undef

  %li32_2 = load i32, i32* undef
  %sext_2 = sext i32 %li32_2 to i64
  mul i64 %sext_2, undef

  ; Zero-extended loads are *not* folded
  %li16_2 = load i16, i16* undef
  %zext_0 = zext i16 %li16_2 to i32
  mul i32 %zext_0, undef

  ; Loads with multiple uses are *not* folded
  %li16_3 = load i16, i16* undef
  %sext_3 = sext i16 %li16_3 to i32
  %sext_4 = sext i16 %li16_3 to i32
  mul i32 %sext_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = mul i32 %tr, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li16_0 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i16 %li16_0 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = mul i32 %sext_0, undef
; Z13:   Cost Model: Found an estimated cost of 1 for instruction:   %li16_1 = load i16, i16* undef
; Z14:   Cost Model: Found an estimated cost of 0 for instruction:   %li16_1 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i16 %li16_1 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = mul i64 %sext_1, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = mul i64 %sext_2, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_2 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %zext_0 = zext i16 %li16_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %9 = mul i32 %zext_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li16_3 = load i16, i16* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_3 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_4 = sext i16 %li16_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %10 = mul i32 %sext_3, undef
}

define void @mul_i16_mem16(i16 %Arg, i16* %Src1, i16* %Src2, i16* %Dst, i32* %Src32) {
  %L1 = load i16, i16* %Src1
  %P0 = mul i16 %Arg, %L1
  store volatile i16 %P0, i16* %Dst

  %L2 = load i16, i16* %Src1
  %L3 = load i16, i16* %Src2
  %P1 = mul i16 %L2, %L3
  store volatile i16 %P1, i16* %Dst

  ; Truncated load
  %L32 = load i32, i32* %Src32
  %tr = trunc i32 %L32 to i16
  %P2 = mul i16 %Arg, %tr
  store volatile i16 %P2, i16* %Dst

  ret void
; CHECK: function 'mul_i16_mem16'
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L1 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %P0 = mul i16 %Arg, %L1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %P0, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L2 = load i16, i16* %Src1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %L3 = load i16, i16* %Src2
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %P1 = mul i16 %L2, %L3
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %P1, i16* %Dst
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %L32 = load i32, i32* %Src32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i32 %L32 to i16
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %P2 = mul i16 %Arg, %tr
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   store volatile i16 %P2, i16* %Dst
}

define void @sdiv_lhs(i32 %arg32, i64 %arg64) {
  %li32 = load i32, i32* undef
  sdiv i32 %li32, %arg32

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  sdiv i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  sdiv i64 %li64, %arg64

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  sdiv i64 %li64_0, %li64_1

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  sdiv i32 %tr, undef

  ; Sign-extended loads
  %li32_2 = load i32, i32* undef
  %sext_0 = sext i32 %li32_2 to i64
  sdiv i64 %sext_0, undef

  ; Loads with multiple uses are *not* folded
  %li32_3 = load i32, i32* undef
  %sext_1 = sext i32 %li32_3 to i64
  %sext_2 = sext i32 %li32_3 to i64
  sdiv i64 %sext_1, undef

  ret void;

; An sdiv loaded dividend (lhs) operand is *not* foldable.
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %1 = sdiv i32 %li32, %arg32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %2 = sdiv i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %3 = sdiv i64 %li64, %arg64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %4 = sdiv i64 %li64_0, %li64_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %5 = sdiv i32 %tr, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %6 = sdiv i64 %sext_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %7 = sdiv i64 %sext_1, undef
}

define void @sdiv_rhs(i32 %arg32, i64 %arg64) {
  %li32 = load i32, i32* undef
  sdiv i32 %arg32, %li32

  %li64 = load i64, i64* undef
  sdiv i64 %arg64, %li64

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr = trunc i64 %li64_2 to i32
  sdiv i32 undef, %tr

  ; Sign-extended loads
  %li32_2 = load i32, i32* undef
  %sext_0 = sext i32 %li32_2 to i64
  sdiv i64 undef, %sext_0

  ; Loads with multiple uses are *not* folded
  %li32_3 = load i32, i32* undef
  %sext_1 = sext i32 %li32_3 to i64
  %sext_2 = sext i32 %li32_3 to i64
  sdiv i64 undef, %sext_1

  ret void;

; An sdiv loaded divisor (rhs) operand is foldable.
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %1 = sdiv i32 %arg32, %li32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %2 = sdiv i64 %arg64, %li64
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %3 = sdiv i32 undef, %tr
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_0 = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %4 = sdiv i64 undef, %sext_0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_1 = sext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext_2 = sext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %5 = sdiv i64 undef, %sext_1
}

define void @udiv_lhs(i32 %arg32, i64 %arg64) {
  %li32 = load i32, i32* undef
  udiv i32 %li32, %arg32

  %li32_0 = load i32, i32* undef
  %li32_1 = load i32, i32* undef
  udiv i32 %li32_0, %li32_1

  %li64 = load i64, i64* undef
  udiv i64 %li64, %arg64

  %li64_0 = load i64, i64* undef
  %li64_1 = load i64, i64* undef
  udiv i64 %li64_0, %li64_1

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  udiv i32 %tr_0, undef

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  udiv i64 %li64_3, undef

  ret void;

; An udiv loaded dividend (lhs) operand is *not* foldable.
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %1 = udiv i32 %li32, %arg32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li32_0 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_1 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %2 = udiv i32 %li32_0, %li32_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %3 = udiv i64 %li64, %arg64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_0 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_1 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %4 = udiv i64 %li64_0, %li64_1
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %5 = udiv i32 %tr_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 10 for instruction:   %6 = udiv i64 %li64_3, undef
}

define void @udiv_rhs(i32 %arg32, i64 %arg64) {
  %li32 = load i32, i32* undef
  udiv i32 %arg32, %li32

  %li64 = load i64, i64* undef
  udiv i64 %arg64, %li64

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  udiv i32 undef, %tr_0

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  udiv i64 undef, %li64_3

  ret void;

; An udiv loaded divisor (rhs) operand is foldable.
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %1 = udiv i32 %arg32, %li32
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %2 = udiv i64 %arg64, %li64
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %3 = udiv i32 undef, %tr_0
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 20 for instruction:   %4 = udiv i64 undef, %li64_3
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  and i32 %tr_0, undef

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  and i64 %li64_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = and i32 %tr_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = and i64 %li64_3, undef
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  or i32 %tr_0, undef

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  or i64 %li64_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = or i32 %tr_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = or i64 %li64_3, undef
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  xor i32 %tr_0, undef

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  xor i64 %li64_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = xor i32 %tr_0, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = xor i64 %li64_3, undef
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

  ; Truncated load
  %li64_2 = load i64, i64* undef
  %tr_0 = trunc i64 %li64_2 to i32
  icmp eq i32 %tr_0, undef

  ; Sign-extended load
  %li32_2 = load i32, i32* undef
  %sext = sext i32 %li32_2 to i64
  icmp eq i64 %sext, undef

  ; Zero-extended load
  %li32_3 = load i32, i32* undef
  %zext = zext i32 %li32_3 to i64
  icmp eq i64 %zext, undef

  ; Loads with multiple uses are *not* folded
  %li64_3 = load i64, i64* undef
  %tr_1 = trunc i64 %li64_3 to i32
  icmp eq i64 %li64_3, undef

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
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li64_2 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_0 = trunc i64 %li64_2 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %5 = icmp eq i32 %tr_0, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_2 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %sext = sext i32 %li32_2 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %6 = icmp eq i64 %sext, undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %li32_3 = load i32, i32* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %zext = zext i32 %li32_3 to i64
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %7 = icmp eq i64 %zext, undef
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %li64_3 = load i64, i64* undef
; CHECK: Cost Model: Found an estimated cost of 0 for instruction:   %tr_1 = trunc i64 %li64_3 to i32
; CHECK: Cost Model: Found an estimated cost of 1 for instruction:   %8 = icmp eq i64 %li64_3, undef
}
