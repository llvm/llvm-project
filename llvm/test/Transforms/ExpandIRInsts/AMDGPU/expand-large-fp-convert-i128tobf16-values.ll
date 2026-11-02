; RUN: opt -S -mtriple=amdgpu9.00-amd-amdhsa \
; RUN:   -passes='require<libcall-lowering-info>,expand-ir-insts,simplifycfg,instcombine' \
; RUN:   < %s | FileCheck %s

; Regression tests for the double rounding of wide integer to bfloat
; conversions (issue #234750).
;
; expand-ir-insts expands i128 to bfloat on AMDGPU. The expansion used to round
; the integer to a 24-bit float intermediate and then fptrunc it to bfloat; for
; integers with more than 24 significant bits the two roundings could disagree
; with a single correctly rounded conversion, leaving the result one ulp off.
; The expansion now assembles the bfloat bit pattern directly.
;
; The expansion only produces integer instructions, so let instcombine fold the
; expansion and check the resulting value. The expansion is also checked
; structurally in X86/expand-large-fp-convert-ui129tobf16.ll.
;
; The values marked "above a midpoint" were miscompiled before the fix; the
; rest guard the exact ties and the overflow boundary.

; 2^30 + 2^22 + 1 is 1 above a bfloat midpoint, so it must round up. The old
; expansion dropped the 1 and rounded the midpoint to even, giving 0x4E80.
define i32 @ui128_above_midpoint() {
; CHECK-LABEL: define i32 @ui128_above_midpoint(
; CHECK:       ret i32 20097
  %r = uitofp i128 1077936129 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; The same value negated: the rounding must apply to the magnitude.
define i32 @si128_above_midpoint_negative() {
; CHECK-LABEL: define i32 @si128_above_midpoint_negative(
; CHECK:       ret i32 52865
  %r = sitofp i128 -1077936129 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; The exact midpoint, 2^30 + 2^22, rounds to even (0x4E80), not up.
define i32 @ui128_exact_midpoint() {
; CHECK-LABEL: define i32 @ui128_exact_midpoint(
; CHECK:       ret i32 20096
  %r = uitofp i128 1077936128 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^100 + 2^92 + 1 is 1 above a midpoint of a much larger exponent; the old
; expansion gave 0x7180.
define i32 @ui128_large_exponent_above_midpoint() {
; CHECK-LABEL: define i32 @ui128_large_exponent_above_midpoint(
; CHECK:       ret i32 29057
  %r = uitofp i128 1272602360385370922596299702273 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; 2^24 is exactly representable in bfloat and must stay exact.
define i32 @ui128_exact() {
; CHECK-LABEL: define i32 @ui128_exact(
; CHECK:       ret i32 19328
  %r = uitofp i128 16777216 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; Largest power of two that is still finite.
define i32 @ui128_two_pow_127() {
; CHECK-LABEL: define i32 @ui128_two_pow_127(
; CHECK:       ret i32 32512
  %r = uitofp i128 170141183460469231731687303715884105728 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; i128 has no exponent-saturation select, so the all-ones significand must
; carry into infinity on its own.
define i32 @ui128_max_rounds_to_inf() {
; CHECK-LABEL: define i32 @ui128_max_rounds_to_inf(
; CHECK:       ret i32 32640
  %r = uitofp i128 340282366920938463463374607431768211455 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}

; INT_MIN of i128: exact power of two, sign must be preserved.
define i32 @si128_min() {
; CHECK-LABEL: define i32 @si128_min(
; CHECK:       ret i32 65280
  %r = sitofp i128 -170141183460469231731687303715884105728 to bfloat
  %bits = bitcast bfloat %r to i16
  %result = zext i16 %bits to i32
  ret i32 %result
}
