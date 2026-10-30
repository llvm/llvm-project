; RUN: opt -S -mtriple=x86_64-- \
; RUN:   -passes='require<libcall-lowering-info>,expand-ir-insts,simplifycfg,instcombine' \
; RUN:   < %s | FileCheck %s

; Value-level regression tests for issue #234750: uitofp/sitofp of an integer
; wider than the target supports to bfloat used to round through float first,
; so the result could be off by one ulp.
;
; x86-64 only expands integer to float conversions wider than 128 bits, so i129
; is used to force the expansion. The expansion only produces integer
; instructions; instcombine folds it so that the resulting bfloat can be
; checked through its bit pattern. The expansion itself is checked structurally
; in expand-large-fp-convert-ui129tobf16.ll, and the i128 bfloat case (which
; AMDGPU does expand) in AMDGPU/expand-large-fp-convert-i128tobf16-values.ll.

; The double-rounding witness from the issue. The exact value is 0x4FF3; the
; round-to-nearest-through-float expansion used to give 0x4FF2.
define i32 @ui129_issue_witness() {
; CHECK-LABEL: define i32 @ui129_issue_witness(
; CHECK:       ret i32 20467
  %r = uitofp i129 8136949792 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; Keep the same midpoint+1 regression at a width substantially larger than the
; x86 integer conversion limit.
define i32 @ui256_above_midpoint() {
; CHECK-LABEL: define i32 @ui256_above_midpoint(
; CHECK:       ret i32 29057
  %r = uitofp i256 1272602360385370922596299702273 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

define i32 @si129_negative_witness() {
; CHECK-LABEL: define i32 @si129_negative_witness(
; CHECK:       ret i32 53235
  %r = sitofp i129 -8136949792 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^30 + 2^22 + 1 is 1 above a bfloat midpoint. The discarded 1 must round it
; up; dropping the 1 would round the exact midpoint to even instead.
define i32 @ui129_above_midpoint() {
; CHECK-LABEL: define i32 @ui129_above_midpoint(
; CHECK:       ret i32 20097
  %r = uitofp i129 1077936129 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; Exact power of two stays exact.
define i32 @ui129_exact_power_of_two() {
; CHECK-LABEL: define i32 @ui129_exact_power_of_two(
; CHECK:       ret i32 20096
  %r = uitofp i129 1073741824 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 511 = 2^9 - 1 is a tie that rounds up into the next exponent, producing 512;
; this exercises the significand carry-out path (itofp-if-then20).
define i32 @ui129_carry_out() {
; CHECK-LABEL: define i32 @ui129_carry_out(
; CHECK:       ret i32 17408
  %r = uitofp i129 511 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; INT_MIN for signed i129.
define i32 @si129_min() {
; CHECK-LABEL: define i32 @si129_min(
; CHECK:       ret i32 65408
  %r = sitofp i129 -340282366920938463463374607431768211456 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; The negative counterpart of the overflow tie: the significand rounds up into
; the next exponent, producing -inf without the exponent saturation select.
define i32 @si129_overflow_tie() {
; CHECK-LABEL: define i32 @si129_overflow_tie(
; CHECK:       ret i32 65408
  %r = sitofp i129 -339617752923046005526922703901628039168 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; Largest finite bfloat; one more rounds to infinity. This is a second
; double-rounding witness: the old expansion produced infinity here.
define i32 @ui129_bf16_max_finite() {
; CHECK-LABEL: define i32 @ui129_bf16_max_finite(
; CHECK:       ret i32 32639
  %r = uitofp i129 339617752923046005526922703901628039167 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^128 - 2^119 is the tie that rounds to infinity.
define i32 @ui129_bf16_overflow_tie() {
; CHECK-LABEL: define i32 @ui129_bf16_overflow_tie(
; CHECK:       ret i32 32640
  %r = uitofp i129 339617752923046005526922703901628039168 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; Largest power of two that is still finite: the exponent saturation select
; must not fire for exponent 127.
define i32 @ui129_two_pow_127() {
; CHECK-LABEL: define i32 @ui129_two_pow_127(
; CHECK:       ret i32 32512
  %r = uitofp i129 170141183460469231731687303715884105728 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^128 does not fit the bfloat exponent; the saturation select must yield inf.
define i32 @ui129_two_pow_128() {
; CHECK-LABEL: define i32 @ui129_two_pow_128(
; CHECK:       ret i32 32640
  %r = uitofp i129 340282366920938463463374607431768211456 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^128 + 1 has a discarded low bit; it still saturates to infinity.
define i32 @ui129_two_pow_128_plus_one() {
; CHECK-LABEL: define i32 @ui129_two_pow_128_plus_one(
; CHECK:       ret i32 32640
  %r = uitofp i129 340282366920938463463374607431768211457 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^128 - 1: the significand rounds up into the next exponent, which is
; infinity (the select only covers exponent >= 128).
define i32 @ui129_all_ones_rounds_to_inf() {
; CHECK-LABEL: define i32 @ui129_all_ones_rounds_to_inf(
; CHECK:       ret i32 32640
  %r = uitofp i129 340282366920938463463374607431768211455 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}
