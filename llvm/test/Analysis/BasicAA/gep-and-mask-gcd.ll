; RUN: opt %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; Trailing zeros from `and` masking should boost the GCD in BasicAA's
; modular analysis, matching the precision that `shl` already achieves.

; Baseline: shl decomposes into Scale=8 via GetLinearExpression.
; GCD=8 (from countr_zero alone). Offset 4 mod 8 = 4 >= 4. NoAlias.
; CHECK-LABEL: Function: shl_noalias
; CHECK: NoAlias: i32* %p, i32* %q
define void @shl_noalias(ptr %base, i64 %x) {
  %even = shl i64 %x, 1
  %p = getelementptr i32, ptr %base, i64 %even
  %q = getelementptr i32, ptr %base, i64 1
  store i32 0, ptr %p
  %v = load i32, ptr %q
  ret void
}

; and clears bit 0 -> V has 1 trailing zero. Scale=4, so product has
; 1+2=3 trailing zeros -> GCD=8. Offset 4 mod 8 = 4 >= 4. NoAlias.
; CHECK-LABEL: Function: and_clear_bit0
; CHECK: NoAlias: i32* %p, i32* %q
define void @and_clear_bit0(ptr %base, i64 %x) {
  %even = and i64 %x, -2
  %p = getelementptr i32, ptr %base, i64 %even
  %q = getelementptr i32, ptr %base, i64 1
  store i32 0, ptr %p
  %v = load i32, ptr %q
  ret void
}

; and clears 2 low bits -> 2 trailing zeros. Scale=4, 2+2=4 -> GCD=16.
; Offset 4 mod 16 = 4 >= 4. NoAlias.
; CHECK-LABEL: Function: and_clear_2bits
; CHECK: NoAlias: i32* %p, i32* %q
define void @and_clear_2bits(ptr %base, i64 %x) {
  %aligned = and i64 %x, -4
  %p = getelementptr i32, ptr %base, i64 %aligned
  %q = getelementptr i32, ptr %base, i64 1
  store i32 0, ptr %p
  %v = load i32, ptr %q
  ret void
}

; and clears bit 0, but accessed element is at offset 0 from base.
; GCD=8, ModOffset = 0 mod 8 = 0, 0 < 4. MayAlias (correctly).
; CHECK-LABEL: Function: and_same_offset
; CHECK: MayAlias: i32* %base, i32* %p
define void @and_same_offset(ptr %base, i64 %x) {
  %even = and i64 %x, -2
  %p = getelementptr i32, ptr %base, i64 %even
  store i32 0, ptr %base
  %v = load i32, ptr %p
  ret void
}

; Negative: no trailing zeros in the mask (clears top bit only).
; Scale=4, GCD=4, offset 4 mod 4 = 0 < 4. MayAlias.
; CHECK-LABEL: Function: and_no_trailing_zeros
; CHECK: MayAlias: i32* %p, i32* %q
define void @and_no_trailing_zeros(ptr %base, i64 %x) {
  %masked = and i64 %x, 9223372036854775807
  %p = getelementptr i32, ptr %base, i64 %masked
  %q = getelementptr i32, ptr %base, i64 1
  store i32 0, ptr %p
  %v = load i32, ptr %q
  ret void
}

; i8 element type: Scale=1, trailing zeros from and = 1. GCD=2.
; Offset 1 mod 2 = 1 >= 1. NoAlias.
; CHECK-LABEL: Function: and_i8_element
; CHECK: NoAlias: i8* %p, i8* %q
define void @and_i8_element(ptr %base, i64 %x) {
  %even = and i64 %x, -2
  %p = getelementptr i8, ptr %base, i64 %even
  %q = getelementptr i8, ptr %base, i64 1
  store i8 0, ptr %p
  %v = load i8, ptr %q
  ret void
}

; Negative: without inbounds, IsNSW=false, Scale=3, ScaleForGCD=2^tz(3)=1.
; Shifted to 1<<1=2, GCD=2, ModOffset=2 mod 2=0 < 2. MayAlias.
; CHECK-LABEL: Function: and_no_inbounds
; CHECK: MayAlias: i16* %p, i16* %q
define void @and_no_inbounds(ptr %base, i64 %x) {
  %even = and i64 %x, -2
  %p = getelementptr [3 x i8], ptr %base, i64 %even
  %q = getelementptr i8, ptr %base, i64 2
  store i16 0, ptr %p
  %v = load i16, ptr %q
  ret void
}

; With inbounds: IsNSW=true, Scale=3 from [3 x i8]. VarTZ=1 from and.
; ScaleForGCD=|3|<<1=6, GCD=6, ModOffset=2 mod 6=2, 6-2=4>=2. NoAlias.
; CHECK-LABEL: Function: and_inbounds
; CHECK: NoAlias: i16* %p, i16* %q
define void @and_inbounds(ptr %base, i64 %x) {
  %even = and i64 %x, -2
  %p = getelementptr inbounds [3 x i8], ptr %base, i64 %even
  %q = getelementptr inbounds i8, ptr %base, i64 2
  store i16 0, ptr %p
  %v = load i16, ptr %q
  ret void
}
