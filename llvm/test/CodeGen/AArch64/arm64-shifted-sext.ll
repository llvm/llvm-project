; RUN: llc < %s -mtriple=arm64-apple-ios | FileCheck %s
;
; <rdar://problem/13820218>

define signext i16 @extendedLeftShiftcharToshortBy4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharToshortBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfiz w0, [[REG]], #4, #8
  %inc = add i8 %a, 1
  %conv1 = sext i8 %inc to i32
  %shl = shl nsw i32 %conv1, 4
  %conv2 = trunc i32 %shl to i16
  ret i16 %conv2
}

define signext i16 @extendedRightShiftcharToshortBy4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharToshortBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfx w0, [[REG]], #4, #4
  %inc = add i8 %a, 1
  %conv1 = sext i8 %inc to i32
  %shr4 = lshr i32 %conv1, 4
  %conv2 = trunc i32 %shr4 to i16
  ret i16 %conv2
}

define signext i16 @extendedLeftShiftcharToshortBy8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharToshortBy8:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfiz w0, [[REG]], #8, #8
  %inc = add i8 %a, 1
  %conv1 = sext i8 %inc to i32
  %shl = shl nsw i32 %conv1, 8
  %conv2 = trunc i32 %shl to i16
  ret i16 %conv2
}

define signext i16 @extendedRightShiftcharToshortBy8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharToshortBy8:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sxtb [[REG]], [[REG]]
; CHECK: asr w0, [[REG]], #8
  %inc = add i8 %a, 1
  %conv1 = sext i8 %inc to i32
  %shr4 = lshr i32 %conv1, 8
  %conv2 = trunc i32 %shr4 to i16
  ret i16 %conv2
}

define i32 @extendedLeftShiftcharTointBy4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharTointBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfiz w0, [[REG]], #4, #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i32
  %shl = shl nsw i32 %conv, 4
  ret i32 %shl
}

define i32 @extendedRightShiftcharTointBy4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharTointBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfx w0, [[REG]], #4, #4
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i32
  %shr = ashr i32 %conv, 4
  ret i32 %shr
}

define i32 @extendedLeftShiftcharTointBy8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharTointBy8:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfiz w0, [[REG]], #8, #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i32
  %shl = shl nsw i32 %conv, 8
  ret i32 %shl
}

define i32 @extendedRightShiftcharTointBy8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharTointBy8:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sxtb [[REG]], [[REG]]
; CHECK: asr w0, [[REG]], #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i32
  %shr = ashr i32 %conv, 8
  ret i32 %shr
}

define i64 @extendedLeftShiftcharToint64By4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfiz x0, x[[REG]], #4, #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i64
  %shl = shl nsw i64 %conv, 4
  ret i64 %shl
}

define i64 @extendedRightShiftcharToint64By4(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfx x0, x[[REG]], #4, #4
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i64
  %shr = ashr i64 %conv, 4
  ret i64 %shr
}

define i64 @extendedLeftShiftcharToint64By8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftcharToint64By8:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfiz x0, x[[REG]], #8, #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i64
  %shl = shl nsw i64 %conv, 8
  ret i64 %shl
}

define i64 @extendedRightShiftcharToint64By8(i8 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftcharToint64By8:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sxtb x[[REG]], w[[REG]]
; CHECK: asr x0, x[[REG]], #8
  %inc = add i8 %a, 1
  %conv = sext i8 %inc to i64
  %shr = ashr i64 %conv, 8
  ret i64 %shr
}

define i32 @extendedLeftShiftshortTointBy4(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftshortTointBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfiz w0, [[REG]], #4, #16
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i32
  %shl = shl nsw i32 %conv, 4
  ret i32 %shl
}

define i32 @extendedRightShiftshortTointBy4(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftshortTointBy4:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sbfx w0, [[REG]], #4, #12
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i32
  %shr = ashr i32 %conv, 4
  ret i32 %shr
}

define i32 @extendedLeftShiftshortTointBy16(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftshortTointBy16:
; CHECK: lsl [[REG:w[0-9]+]], w0, #16
; CHECK: add w0, [[REG]], #16, lsl #12
  %inc = add i16 %a, 1
  %conv2 = zext i16 %inc to i32
  %shl = shl nuw i32 %conv2, 16
  ret i32 %shl
}

define i32 @extendedRightShiftshortTointBy16(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftshortTointBy16:
; CHECK: add [[REG:w[0-9]+]], w0, #1
; CHECK: sxth [[REG]], [[REG]]
; CHECK: asr w0, [[REG]], #16
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i32
  %shr = ashr i32 %conv, 16
  ret i32 %shr
}

define i64 @extendedLeftShiftshortToint64By4(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftshortToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfiz x0, x[[REG]], #4, #16
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i64
  %shl = shl nsw i64 %conv, 4
  ret i64 %shl
}

define i64 @extendedRightShiftshortToint64By4(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftshortToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfx x0, x[[REG]], #4, #12
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i64
  %shr = ashr i64 %conv, 4
  ret i64 %shr
}

define i64 @extendedLeftShiftshortToint64By16(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftshortToint64By16:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfiz x0, x[[REG]], #16, #16
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i64
  %shl = shl nsw i64 %conv, 16
  ret i64 %shl
}

define i64 @extendedRightShiftshortToint64By16(i16 signext %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftshortToint64By16:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sxth x[[REG]], w[[REG]]
; CHECK: asr x0, x[[REG]], #16
  %inc = add i16 %a, 1
  %conv = sext i16 %inc to i64
  %shr = ashr i64 %conv, 16
  ret i64 %shr
}

define i64 @extendedLeftShiftintToint64By4(i32 %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftintToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfiz x0, x[[REG]], #4, #32
  %inc = add nsw i32 %a, 1
  %conv = sext i32 %inc to i64
  %shl = shl nsw i64 %conv, 4
  ret i64 %shl
}

define i64 @extendedRightShiftintToint64By4(i32 %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftintToint64By4:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sbfx x0, x[[REG]], #4, #28
  %inc = add nsw i32 %a, 1
  %conv = sext i32 %inc to i64
  %shr = ashr i64 %conv, 4
  ret i64 %shr
}

define i64 @extendedLeftShiftintToint64By32(i32 %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedLeftShiftintToint64By32:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: lsl x0, x[[REG]], #32
  %inc = add nsw i32 %a, 1
  %conv2 = zext i32 %inc to i64
  %shl = shl nuw i64 %conv2, 32
  ret i64 %shl
}

define i64 @extendedRightShiftintToint64By32(i32 %a) nounwind readnone ssp {
entry:
; CHECK-LABEL: extendedRightShiftintToint64By32:
; CHECK: add w[[REG:[0-9]+]], w0, #1
; CHECK: sxtw x[[REG]], w[[REG]]
; CHECK: asr x0, x[[REG]], #32
  %inc = add nsw i32 %a, 1
  %conv = sext i32 %inc to i64
  %shr = ashr i64 %conv, 32
  ret i64 %shr
}
