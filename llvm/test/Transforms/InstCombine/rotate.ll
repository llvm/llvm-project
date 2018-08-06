; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; These are UB-free rotate left/right patterns that are narrowed to a smaller bitwidth.
; See PR34046 and PR16726 for motivating examples:
; https://bugs.llvm.org/show_bug.cgi?id=34046
; https://bugs.llvm.org/show_bug.cgi?id=16726

define i16 @rotate_left_16bit(i16 %v, i32 %shift) {
; CHECK-LABEL: @rotate_left_16bit(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %shift to i16
; CHECK-NEXT:    [[TMP2:%.*]] = and i16 [[TMP1]], 15
; CHECK-NEXT:    [[TMP3:%.*]] = sub i16 0, [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = and i16 [[TMP3]], 15
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i16 %v, [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = shl i16 %v, [[TMP2]]
; CHECK-NEXT:    [[CONV2:%.*]] = or i16 [[TMP5]], [[TMP6]]
; CHECK-NEXT:    ret i16 [[CONV2]]
;
  %and = and i32 %shift, 15
  %conv = zext i16 %v to i32
  %shl = shl i32 %conv, %and
  %sub = sub i32 16, %and
  %shr = lshr i32 %conv, %sub
  %or = or i32 %shr, %shl
  %conv2 = trunc i32 %or to i16
  ret i16 %conv2
}

; Commute the 'or' operands and try a vector type.

define <2 x i16> @rotate_left_commute_16bit_vec(<2 x i16> %v, <2 x i32> %shift) {
; CHECK-LABEL: @rotate_left_commute_16bit_vec(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc <2 x i32> %shift to <2 x i16>
; CHECK-NEXT:    [[TMP2:%.*]] = and <2 x i16> [[TMP1]], <i16 15, i16 15>
; CHECK-NEXT:    [[TMP3:%.*]] = sub <2 x i16> zeroinitializer, [[TMP1]]
; CHECK-NEXT:    [[TMP4:%.*]] = and <2 x i16> [[TMP3]], <i16 15, i16 15>
; CHECK-NEXT:    [[TMP5:%.*]] = shl <2 x i16> %v, [[TMP2]]
; CHECK-NEXT:    [[TMP6:%.*]] = lshr <2 x i16> %v, [[TMP4]]
; CHECK-NEXT:    [[CONV2:%.*]] = or <2 x i16> [[TMP5]], [[TMP6]]
; CHECK-NEXT:    ret <2 x i16> [[CONV2]]
;
  %and = and <2 x i32> %shift, <i32 15, i32 15>
  %conv = zext <2 x i16> %v to <2 x i32>
  %shl = shl <2 x i32> %conv, %and
  %sub = sub <2 x i32> <i32 16, i32 16>, %and
  %shr = lshr <2 x i32> %conv, %sub
  %or = or <2 x i32> %shl, %shr
  %conv2 = trunc <2 x i32> %or to <2 x i16>
  ret <2 x i16> %conv2
}

; Change the size, rotation direction (the subtract is on the left-shift), and mask op.

define i8 @rotate_right_8bit(i8 %v, i3 %shift) {
; CHECK-LABEL: @rotate_right_8bit(
; CHECK-NEXT:    [[TMP1:%.*]] = zext i3 %shift to i8
; CHECK-NEXT:    [[TMP2:%.*]] = sub i3 0, %shift
; CHECK-NEXT:    [[TMP3:%.*]] = zext i3 [[TMP2]] to i8
; CHECK-NEXT:    [[TMP4:%.*]] = shl i8 %v, [[TMP3]]
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i8 %v, [[TMP1]]
; CHECK-NEXT:    [[CONV2:%.*]] = or i8 [[TMP4]], [[TMP5]]
; CHECK-NEXT:    ret i8 [[CONV2]]
;
  %and = zext i3 %shift to i32
  %conv = zext i8 %v to i32
  %shr = lshr i32 %conv, %and
  %sub = sub i32 8, %and
  %shl = shl i32 %conv, %sub
  %or = or i32 %shl, %shr
  %conv2 = trunc i32 %or to i8
  ret i8 %conv2
}

; The shifted value does not need to be a zexted value; here it is masked.
; The shift mask could be less than the bitwidth, but this is still ok.

define i8 @rotate_right_commute_8bit(i32 %v, i32 %shift) {
; CHECK-LABEL: @rotate_right_commute_8bit(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %shift to i8
; CHECK-NEXT:    [[TMP2:%.*]] = and i8 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = sub nsw i8 0, [[TMP2]]
; CHECK-NEXT:    [[TMP4:%.*]] = and i8 [[TMP3]], 7
; CHECK-NEXT:    [[TMP5:%.*]] = trunc i32 %v to i8
; CHECK-NEXT:    [[TMP6:%.*]] = lshr i8 [[TMP5]], [[TMP2]]
; CHECK-NEXT:    [[TMP7:%.*]] = shl i8 [[TMP5]], [[TMP4]]
; CHECK-NEXT:    [[CONV2:%.*]] = or i8 [[TMP6]], [[TMP7]]
; CHECK-NEXT:    ret i8 [[CONV2]]
;
  %and = and i32 %shift, 3
  %conv = and i32 %v, 255
  %shr = lshr i32 %conv, %and
  %sub = sub i32 8, %and
  %shl = shl i32 %conv, %sub
  %or = or i32 %shr, %shl
  %conv2 = trunc i32 %or to i8
  ret i8 %conv2
}

; If the original source does not mask the shift amount,
; we still do the transform by adding masks to make it safe.

define i8 @rotate8_not_safe(i8 %v, i32 %shamt) {
; CHECK-LABEL: @rotate8_not_safe(
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i32 %shamt to i8
; CHECK-NEXT:    [[TMP2:%.*]] = sub i8 0, [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = and i8 [[TMP1]], 7
; CHECK-NEXT:    [[TMP4:%.*]] = and i8 [[TMP2]], 7
; CHECK-NEXT:    [[TMP5:%.*]] = lshr i8 %v, [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = shl i8 %v, [[TMP3]]
; CHECK-NEXT:    [[RET:%.*]] = or i8 [[TMP5]], [[TMP6]]
; CHECK-NEXT:    ret i8 [[RET]]
;
  %conv = zext i8 %v to i32
  %sub = sub i32 8, %shamt
  %shr = lshr i32 %conv, %sub
  %shl = shl i32 %conv, %shamt
  %or = or i32 %shr, %shl
  %ret = trunc i32 %or to i8
  ret i8 %ret
}

; The next two tests mask sure we narrower (x << (x & 15)) | (x >> (-x & 15))
; when types have been promoted.
; FIXME: We should be able to narrow this.

define i16 @rotate16_neg_mask(i16 %v, i16 %shamt) {
; CHECK-LABEL: @rotate16_neg_mask(
; CHECK-NEXT:    [[CONV:%.*]] = zext i16 [[V:%.*]] to i32
; CHECK-NEXT:    [[RSHAMT:%.*]] = and i16 [[SHAMT:%.*]], 15
; CHECK-NEXT:    [[RSHAMTCONV:%.*]] = zext i16 [[RSHAMT]] to i32
; CHECK-NEXT:    [[SHR:%.*]] = lshr i32 [[CONV]], [[RSHAMTCONV]]
; CHECK-NEXT:    [[NEG:%.*]] = sub i16 0, [[SHAMT]]
; CHECK-NEXT:    [[LSHAMT:%.*]] = and i16 [[NEG]], 15
; CHECK-NEXT:    [[LSHAMTCONV:%.*]] = zext i16 [[LSHAMT]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[LSHAMTCONV]]
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SHR]], [[SHL]]
; CHECK-NEXT:    [[RET:%.*]] = trunc i32 [[OR]] to i16
; CHECK-NEXT:    ret i16 [[RET]]
;
  %conv = zext i16 %v to i32
  %rshamt = and i16 %shamt, 15
  %rshamtconv = zext i16 %rshamt to i32
  %shr = lshr i32 %conv, %rshamtconv
  %neg = sub i16 0, %shamt
  %lshamt = and i16 %neg, 15
  %lshamtconv = zext i16 %lshamt to i32
  %shl = shl i32 %conv, %lshamtconv
  %or = or i32 %shr, %shl
  %ret = trunc i32 %or to i16
  ret i16 %ret
}

define i8 @rotate8_neg_mask(i8 %v, i8 %shamt) {
; CHECK-LABEL: @rotate8_neg_mask(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 [[V:%.*]] to i32
; CHECK-NEXT:    [[RSHAMT:%.*]] = and i8 [[SHAMT:%.*]], 7
; CHECK-NEXT:    [[RSHAMTCONV:%.*]] = zext i8 [[RSHAMT]] to i32
; CHECK-NEXT:    [[SHR:%.*]] = lshr i32 [[CONV]], [[RSHAMTCONV]]
; CHECK-NEXT:    [[NEG:%.*]] = sub i8 0, [[SHAMT]]
; CHECK-NEXT:    [[LSHAMT:%.*]] = and i8 [[NEG]], 7
; CHECK-NEXT:    [[LSHAMTCONV:%.*]] = zext i8 [[LSHAMT]] to i32
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[LSHAMTCONV]]
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SHR]], [[SHL]]
; CHECK-NEXT:    [[RET:%.*]] = trunc i32 [[OR]] to i8
; CHECK-NEXT:    ret i8 [[RET]]
;
  %conv = zext i8 %v to i32
  %rshamt = and i8 %shamt, 7
  %rshamtconv = zext i8 %rshamt to i32
  %shr = lshr i32 %conv, %rshamtconv
  %neg = sub i8 0, %shamt
  %lshamt = and i8 %neg, 7
  %lshamtconv = zext i8 %lshamt to i32
  %shl = shl i32 %conv, %lshamtconv
  %or = or i32 %shr, %shl
  %ret = trunc i32 %or to i8
  ret i8 %ret
}

; The next two types have a shift amount that is already i32 so we would still
; need a truncate for it going into the rotate pattern.
; FIXME: We can narrow this, but we would still need a trunc on the shift amt.

define i16 @rotate16_neg_mask_wide_amount(i16 %v, i32 %shamt) {
; CHECK-LABEL: @rotate16_neg_mask_wide_amount(
; CHECK-NEXT:    [[CONV:%.*]] = zext i16 [[V:%.*]] to i32
; CHECK-NEXT:    [[RSHAMT:%.*]] = and i32 [[SHAMT:%.*]], 15
; CHECK-NEXT:    [[SHR:%.*]] = lshr i32 [[CONV]], [[RSHAMT]]
; CHECK-NEXT:    [[NEG:%.*]] = sub i32 0, [[SHAMT]]
; CHECK-NEXT:    [[LSHAMT:%.*]] = and i32 [[NEG]], 15
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[LSHAMT]]
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SHR]], [[SHL]]
; CHECK-NEXT:    [[RET:%.*]] = trunc i32 [[OR]] to i16
; CHECK-NEXT:    ret i16 [[RET]]
;
  %conv = zext i16 %v to i32
  %rshamt = and i32 %shamt, 15
  %shr = lshr i32 %conv, %rshamt
  %neg = sub i32 0, %shamt
  %lshamt = and i32 %neg, 15
  %shl = shl i32 %conv, %lshamt
  %or = or i32 %shr, %shl
  %ret = trunc i32 %or to i16
  ret i16 %ret
}

define i8 @rotate8_neg_mask_wide_amount(i8 %v, i32 %shamt) {
; CHECK-LABEL: @rotate8_neg_mask_wide_amount(
; CHECK-NEXT:    [[CONV:%.*]] = zext i8 [[V:%.*]] to i32
; CHECK-NEXT:    [[RSHAMT:%.*]] = and i32 [[SHAMT:%.*]], 7
; CHECK-NEXT:    [[SHR:%.*]] = lshr i32 [[CONV]], [[RSHAMT]]
; CHECK-NEXT:    [[NEG:%.*]] = sub i32 0, [[SHAMT]]
; CHECK-NEXT:    [[LSHAMT:%.*]] = and i32 [[NEG]], 7
; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[CONV]], [[LSHAMT]]
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SHR]], [[SHL]]
; CHECK-NEXT:    [[RET:%.*]] = trunc i32 [[OR]] to i8
; CHECK-NEXT:    ret i8 [[RET]]
;
  %conv = zext i8 %v to i32
  %rshamt = and i32 %shamt, 7
  %shr = lshr i32 %conv, %rshamt
  %neg = sub i32 0, %shamt
  %lshamt = and i32 %neg, 7
  %shl = shl i32 %conv, %lshamt
  %or = or i32 %shr, %shl
  %ret = trunc i32 %or to i8
  ret i8 %ret
}
