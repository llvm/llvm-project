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

define i16 @narrow_sext_sub(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_sext_sub(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = sub i16 %x16, [[TMP1]]
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = sext i16 %x16 to i32
  %b = sub i32 %x32, %y32
  %r = trunc i32 %b to i16
  ret i16 %r
}

define i16 @narrow_zext_sub(i16 %x16, i32 %y32) {
; CHECK-LABEL: @narrow_zext_sub(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %y32 to i16
; CHECK-NEXT:    [[R:%.*]] = sub i16 %x16, [[TMP1]]
; CHECK-NEXT:    ret i16 [[R]]
;
  %x32 = zext i16 %x16 to i32
  %b = sub i32 %x32, %y32
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

define <2 x i16> @narrow_sext_sub_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_sext_sub_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = sub <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = sext <2 x i16> %x16 to <2 x i32>
  %b = sub <2 x i32> %y32op0, %x32
  %r = trunc <2 x i32> %b to <2 x i16>
  ret <2 x i16> %r
}

define <2 x i16> @narrow_zext_sub_commute(<2 x i16> %x16, <2 x i32> %y32) {
; CHECK-LABEL: @narrow_zext_sub_commute(
; CHECK-NEXT:    [[Y32OP0:%.*]] = sdiv <2 x i32> %y32, <i32 7, i32 -17>
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> [[Y32OP0]] to <2 x i16>
; CHECK-NEXT:    [[R:%.*]] = sub <2 x i16> [[TMP1]], %x16
; CHECK-NEXT:    ret <2 x i16> [[R]]
;
  %y32op0 = sdiv <2 x i32> %y32, <i32 7, i32 -17>
  %x32 = zext <2 x i16> %x16 to <2 x i32>
  %b = sub <2 x i32> %y32op0, %x32
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

; Test cases for PR43580
define i8 @narrow_zext_ashr_keep_trunc(i8 %i1, i8 %i2) {
; CHECK-LABEL: @narrow_zext_ashr_keep_trunc(
; CHECK-NEXT:    [[I1_EXT:%.*]] = sext i8 [[I1:%.*]] to i16
; CHECK-NEXT:    [[I2_EXT:%.*]] = sext i8 [[I2:%.*]] to i16
; CHECK-NEXT:    [[SUB:%.*]] = add nsw i16 [[I1_EXT]], [[I2_EXT]]
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i16 [[SUB]], 1
; CHECK-NEXT:    [[T:%.*]] = trunc i16 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[T]]
;
  %i1.ext = sext i8 %i1 to i32
  %i2.ext = sext i8 %i2 to i32
  %sub = add nsw i32 %i1.ext, %i2.ext
  %shift = ashr i32 %sub, 1
  %t = trunc i32 %shift to i8
  ret i8 %t
}

define i8 @narrow_zext_ashr_keep_trunc2(i9 %i1, i9 %i2) {
; CHECK-LABEL: @narrow_zext_ashr_keep_trunc2(
; CHECK-NEXT:    [[I1_EXT1:%.*]] = zext i9 [[I1:%.*]] to i16
; CHECK-NEXT:    [[I2_EXT2:%.*]] = zext i9 [[I2:%.*]] to i16
; CHECK-NEXT:    [[SUB:%.*]] = add nuw nsw i16 [[I1_EXT1]], [[I2_EXT2]]
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i16 [[SUB]], 1
; CHECK-NEXT:    [[T:%.*]] = trunc i16 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[T]]
;
  %i1.ext = sext i9 %i1 to i64
  %i2.ext = sext i9 %i2 to i64
  %sub = add nsw i64 %i1.ext, %i2.ext
  %shift = ashr i64 %sub, 1
  %t = trunc i64 %shift to i8
  ret i8 %t
}

define i7 @narrow_zext_ashr_keep_trunc3(i8 %i1, i8 %i2) {
; CHECK-LABEL: @narrow_zext_ashr_keep_trunc3(
; CHECK-NEXT:    [[I1_EXT1:%.*]] = zext i8 [[I1:%.*]] to i14
; CHECK-NEXT:    [[I2_EXT2:%.*]] = zext i8 [[I2:%.*]] to i14
; CHECK-NEXT:    [[SUB:%.*]] = add nuw nsw i14 [[I1_EXT1]], [[I2_EXT2]]
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i14 [[SUB]], 1
; CHECK-NEXT:    [[T:%.*]] = trunc i14 [[TMP1]] to i7
; CHECK-NEXT:    ret i7 [[T]]
;
  %i1.ext = sext i8 %i1 to i64
  %i2.ext = sext i8 %i2 to i64
  %sub = add nsw i64 %i1.ext, %i2.ext
  %shift = ashr i64 %sub, 1
  %t = trunc i64 %shift to i7
  ret i7 %t
}

define <8 x i8> @narrow_zext_ashr_keep_trunc_vector(<8 x i8> %i1, <8 x i8> %i2) {
; CHECK-LABEL: @narrow_zext_ashr_keep_trunc_vector(
; CHECK-NEXT:    [[I1_EXT:%.*]] = sext <8 x i8> [[I1:%.*]] to <8 x i32>
; CHECK-NEXT:    [[I2_EXT:%.*]] = sext <8 x i8> [[I2:%.*]] to <8 x i32>
; CHECK-NEXT:    [[SUB:%.*]] = add nsw <8 x i32> [[I1_EXT]], [[I2_EXT]]
; CHECK-NEXT:    [[TMP1:%.*]] = lshr <8 x i32> [[SUB]], <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
; CHECK-NEXT:    [[T:%.*]] = trunc <8 x i32> [[TMP1]] to <8 x i8>
; CHECK-NEXT:    ret <8 x i8> [[T]]
;
  %i1.ext = sext <8 x i8> %i1 to <8 x i32>
  %i2.ext = sext <8 x i8> %i2 to <8 x i32>
  %sub = add nsw <8 x i32> %i1.ext, %i2.ext
  %shift = ashr <8 x i32> %sub, <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %t = trunc <8 x i32> %shift to <8 x i8>
  ret <8 x i8> %t
}

define i8 @dont_narrow_zext_ashr_keep_trunc(i8 %i1, i8 %i2) {
; CHECK-LABEL: @dont_narrow_zext_ashr_keep_trunc(
; CHECK-NEXT:    [[I1_EXT:%.*]] = sext i8 [[I1:%.*]] to i16
; CHECK-NEXT:    [[I2_EXT:%.*]] = sext i8 [[I2:%.*]] to i16
; CHECK-NEXT:    [[SUB:%.*]] = add nsw i16 [[I1_EXT]], [[I2_EXT]]
; CHECK-NEXT:    [[TMP1:%.*]] = lshr i16 [[SUB]], 1
; CHECK-NEXT:    [[T:%.*]] = trunc i16 [[TMP1]] to i8
; CHECK-NEXT:    ret i8 [[T]]
;
  %i1.ext = sext i8 %i1 to i16
  %i2.ext = sext i8 %i2 to i16
  %sub = add nsw i16 %i1.ext, %i2.ext
  %shift = ashr i16 %sub, 1
  %t = trunc i16 %shift to i8
  ret i8 %t
}
