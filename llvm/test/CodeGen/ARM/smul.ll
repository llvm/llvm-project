; RUN: llc -mtriple=arm-eabi -mcpu=generic %s -o /dev/null
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb--none-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s

@x = weak global i16 0          ; <i16*> [#uses=1]
@y = weak global i16 0          ; <i16*> [#uses=0]

define i32 @f1(i32 %y) {
; CHECK-LABEL: f1:
; CHECK: smulbt
        %tmp = load i16, i16* @x             ; <i16> [#uses=1]
        %tmp1 = add i16 %tmp, 2         ; <i16> [#uses=1]
        %tmp2 = sext i16 %tmp1 to i32           ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp2, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f2(i32 %x, i32 %y) {
; CHECK-LABEL: f2:
; CHECK: smultt
        %tmp1 = ashr i32 %x, 16         ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp3, %tmp1            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f3(i32 %a, i16 %x, i32 %y) {
; CHECK-LABEL: f3:
; CHECK: smlabt
        %tmp = sext i16 %x to i32               ; <i32> [#uses=1]
        %tmp2 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp3 = mul i32 %tmp2, %tmp             ; <i32> [#uses=1]
        %tmp5 = add i32 %tmp3, %a               ; <i32> [#uses=1]
        ret i32 %tmp5
}

define i32 @f4(i32 %a, i32 %x, i32 %y) {
; CHECK-LABEL: f4:
; CHECK: smlatt
        %tmp1 = ashr i32 %x, 16
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp3, %tmp1
        %tmp5 = add i32 %tmp4, %a
        ret i32 %tmp5
}

define i32 @f5(i32 %a, i16 %x, i16 %y) {
; CHECK-LABEL: f5:
; CHECK: smlabb
        %tmp1 = sext i16 %x to i32
        %tmp3 = sext i16 %y to i32
        %tmp4 = mul i32 %tmp3, %tmp1
        %tmp5 = add i32 %tmp4, %a
        ret i32 %tmp5
}

define i32 @f6(i32 %a, i16 %x, i32 %y) {
; CHECK-LABEL: f6:
; CHECK: smlabt
        %tmp1 = sext i16 %x to i32
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp3, %tmp1
        %tmp5 = add i32 %tmp4, %a
        ret i32 %tmp5
}

define i32 @f7(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f7:
; CHECK: smlawb
        %shl = shl i32 %b, 16
        %shr = ashr exact i32 %shl, 16
        %conv = sext i32 %a to i64
        %conv2 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv2, %conv
        %shr49 = lshr i64 %mul, 16
        %conv5 = trunc i64 %shr49 to i32
        %add = add nsw i32 %conv5, %c
        ret i32 %add
}

define i32 @f8(i32 %a, i16 signext %b, i32 %c) {
; CHECK-LABEL: f8:
; CHECK: smlawb
        %conv = sext i32 %a to i64
        %conv1 = sext i16 %b to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr5 = lshr i64 %mul, 16
        %conv2 = trunc i64 %shr5 to i32
        %add = add nsw i32 %conv2, %c
        ret i32 %add
}

define i32 @f9(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f9:
; CHECK: smlawt
        %conv = sext i32 %a to i64
        %shr = ashr i32 %b, 16
        %conv1 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr26 = lshr i64 %mul, 16
        %conv3 = trunc i64 %shr26 to i32
        %add = add nsw i32 %conv3, %c
        ret i32 %add
}

define i32 @f10(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f10:
; CHECK: smulwb
        %shl = shl i32 %b, 16
        %shr = ashr exact i32 %shl, 16
        %conv = sext i32 %a to i64
        %conv2 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv2, %conv
        %shr37 = lshr i64 %mul, 16
        %conv4 = trunc i64 %shr37 to i32
        ret i32 %conv4
}

define i32 @f11(i32 %a, i16 signext %b, i32 %c) {
; CHECK-LABEL: f11:
; CHECK: smulwb
        %conv = sext i32 %a to i64
        %conv1 = sext i16 %b to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr4 = lshr i64 %mul, 16
        %conv2 = trunc i64 %shr4 to i32
        ret i32 %conv2
}

define i32 @f12(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f12:
; CHECK: smulwt
        %conv = sext i32 %a to i64
        %shr = ashr i32 %b, 16
        %conv1 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr25 = lshr i64 %mul, 16
        %conv3 = trunc i64 %shr25 to i32
        ret i32 %conv3
}
