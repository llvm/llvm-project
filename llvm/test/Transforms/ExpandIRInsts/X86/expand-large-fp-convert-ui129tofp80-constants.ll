; Integer constants with the top bit set are printed as negative decimals, so
; the i80 bit patterns below appear as signed values.
; RUN: opt -S -mtriple=x86_64-- \
; RUN:   -passes='require<libcall-lowering-info>,expand-ir-insts,simplifycfg,instcombine' \
; RUN:   < %s | FileCheck %s

; Value-level regression tests for integer to x86_fp80 conversions. x87
; extended precision stores an explicit integer bit, so the old expansion went
; through fp128 and then truncated to fp80, which could round twice and lose
; the low bit. The expansion now assembles the 80-bit pattern directly; these
; tests fold it so that the result is checked through its bit pattern. The
; expansion itself is checked structurally in expand-large-fp-convert-ui129tofp.ll.

; 2^127 + 2^63 + 1 is one above the midpoint between 2^127 and the next fp80
; value, so it must round up. Rounding through fp128 first makes the discarded
; bit a tie, which then rounds back down: the old expansion gave 0x407E8000000000000000.
define i80 @ui129_double_rounding_witness() {
; CHECK-LABEL: @ui129_double_rounding_witness(
; CHECK:       ret i80 304564968028981551955969
;
  %r = uitofp i129 170141183460469231740910675752738881537 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

define i80 @si129_negative_double_rounding_witness() {
; CHECK-LABEL: @si129_negative_double_rounding_witness(
; CHECK:       ret i80 -299897941778333035397119
;
  %r = sitofp i129 -170141183460469231740910675752738881537 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

; 2^127 + 2^63 is exactly halfway between two fp80 values and rounds to even.
define i80 @ui129_tie_rounds_to_even() {
; CHECK-LABEL: @ui129_tie_rounds_to_even(
; CHECK:       ret i80 304564968028981551955968
;
  %r = uitofp i129 170141183460469231740910675752738881536 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

define i80 @si129_negative_tie_rounds_to_even() {
; CHECK-LABEL: @si129_negative_tie_rounds_to_even(
; CHECK:       ret i80 -299897941778333035397120
;
  %r = sitofp i129 -170141183460469231740910675752738881536 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

; Largest unsigned i129: the significand rounds up into the next exponent
; (itofp-if-then20).
define i80 @ui129_carry_into_next_exponent() {
; CHECK-LABEL: @ui129_carry_into_next_exponent(
; CHECK:       ret i80 304583414773055261507584
;
  %r = uitofp i129 340282366920938463463374607431768211455 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

; The top bit of the input must not turn into the sign bit of an unsigned
; conversion. The fp128 round trip used to sign-extend the input here.
define i80 @ui129_unsigned_top_bit() {
; CHECK-LABEL: @ui129_unsigned_top_bit(
; CHECK:       ret i80 304583414773055261507584
;
  %r = uitofp i129 340282366920938463463374607431768211456 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

; Values that fit in the 64-bit significand are exact.
define i80 @ui129_exact_power_of_two() {
; CHECK-LABEL: @ui129_exact_power_of_two(
; CHECK:       ret i80 303384376408264140652544
;
  %r = uitofp i129 9223372036854775808 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}

define i80 @ui129_one() {
; CHECK-LABEL: @ui129_one(
; CHECK:       ret i80 302222231531620438900736
;
  %r = uitofp i129 1 to x86_fp80
  %bits = bitcast x86_fp80 %r to i80
  ret i80 %bits
}
