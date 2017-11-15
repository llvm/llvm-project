; RUN: opt < %s -instcombine -S | FileCheck %s

define i16 @narrow_sext_and(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_and(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = and i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = and i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_and(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_and(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = and i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = and i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_sext_or(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_or(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = or i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = or i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_or(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_or(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = or i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = or i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_sext_xor(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_xor(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = xor i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = xor i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_xor(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_xor(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = xor i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = xor i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_sext_add(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_add(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = add i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = add i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_add(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_add(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = add i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = add i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_sext_mul(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_mul(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = mul i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = mul i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_mul(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_mul(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = mul i16 [[TMP1]], %x16
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = mul i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

; Verify that the commuted patterns work. The div is to ensure that complexity-based
; canonicalization doesn't swap the binop operands. Use vector types to show those work too.

define <2 x i16> @narrow_sext_and_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_and_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = and <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = and <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_and_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_and_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = and <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = and <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_sext_or_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_or_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = or <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = or <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_or_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_or_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = or <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = or <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_sext_xor_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_xor_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = xor <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = xor <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_xor_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_xor_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = xor <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = xor <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_sext_add_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_add_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = add <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = add <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_add_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_add_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = add <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = add <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_sext_mul_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_mul_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = mul <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = mul <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_mul_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_mul_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = mul <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = mul <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

