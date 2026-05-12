; RUN: opt < %s "-passes=print<scalar-evolution>" -disable-output 2>&1 | FileCheck %s

; Check that, when constructing a SCEV for rounding a value up to the nearest
; multiple of a constant, we always use that largest possible value for the
; added constant, even if instcombine has cleared some bits in the IR because
; the input is already known to be a multiple of some smaller value.

define i32 @mul_2_round_to_16_a(i32 %val) {
; CHECK-LABEL: Classifying expressions for: @mul_2_round_to_16_a
; CHECK: -->  (2 * %val)
  %mul = mul i32 %val, 2
; CHECK: (15 + (2 * %val))
  %add = add i32 %mul, 15
; CHECK: (16 * ((15 + (2 * %val)) /u 16))<nuw>
  %round = and i32 %add, -16
  ret i32 %round
}

define i32 @mul_2_round_to_16_b(i32 %val) {
; CHECK-LABEL: Classifying expressions for: @mul_2_round_to_16_b
; CHECK: (2 * %val)
  %mul = mul i32 %val, 2
; CHECK: (14 + (2 * %val))
  %add = add i32 %mul, 14
; CHECK: (16 * ((15 + (2 * %val)) /u 16))<nuw>
  %round = and i32 %add, -16
  ret i32 %round
}

define i32 @mul_4_round_to_16(i32 %val) {
; CHECK-LABEL: Classifying expressions for: @mul_4_round_to_16
; CHECK: (4 * %val)
  %mul = mul i32 %val, 4
; CHECK: (12 + (4 * %val))
  %add = add i32 %mul, 12
; CHECK: (16 * ((15 + (4 * %val)) /u 16))<nuw>
  %round = and i32 %add, -16
  ret i32 %round
}

; M is not a multiple of N, cannot do transformation
define i32 @invalid1(i32 %val) {
; CHECK-LABEL: Classifying expressions for: @invalid1
; CHECK: (3 * %val)
  %mul = mul i32 %val, 3
; CHECK: (12 + (3 * %val))
  %add = add i32 %mul, 12
; CHECK: (16 * ((12 + (3 * %val)) /u 16))<nuw>
  %round = and i32 %add, -16
  ret i32 %round
}

; M is greater then N, cannot do transformation
define i32 @invalid2(i32 %val) {
; CHECK-LABEL: Classifying expressions for: @invalid2
; CHECK: -->  (4 * %val)
  %mul = mul i32 %val, 4
; CHECK: (1 + (4 * %val))
  %add = add i32 %mul, 1
; CHECK: (2 * ((1 + (4 * %val))<nuw><nsw> /u 2))<nuw>
  %round = and i32 %add, -2
  ret i32 %round
}
