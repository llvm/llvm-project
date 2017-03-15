; RUN: llc -mtriple=arm-eabi -mcpu=generic %s -o /dev/null
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumb--none-eabi -mcpu=cortex-a8 %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6t2-none-eabi %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6-none-eabi %s -o - | FileCheck %s -check-prefix=CHECK-THUMBV6

define i32 @f1(i16 %x, i32 %y) {
; CHECK-LABEL: f1:
; CHECK-NOT: sxth
; CHECK: {{smulbt r0, r0, r1|smultb r0, r1, r0}}
; CHECK-THUMBV6-NOT: {{smulbt|smultb}}
        %tmp1 = sext i16 %x to i32
        %tmp2 = ashr i32 %y, 16
        %tmp3 = mul i32 %tmp2, %tmp1
        ret i32 %tmp3
}

define i32 @f2(i32 %x, i32 %y) {
; CHECK-LABEL: f2:
; CHECK: smultt
; CHECK-THUMBV6-NOT: smultt
        %tmp1 = ashr i32 %x, 16
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp3, %tmp1
        ret i32 %tmp4
}

define i32 @f3(i32 %a, i16 %x, i32 %y) {
; CHECK-LABEL: f3:
; CHECK-NOT: sxth
; CHECK: {{smlabt r0, r1, r2, r0|smlatb r0, r2, r1, r0}}
; CHECK-THUMBV6-NOT: {{smlabt|smlatb}}
        %tmp = sext i16 %x to i32
        %tmp2 = ashr i32 %y, 16
        %tmp3 = mul i32 %tmp2, %tmp
        %tmp5 = add i32 %tmp3, %a
        ret i32 %tmp5
}

define i32 @f4(i32 %a, i32 %x, i32 %y) {
; CHECK-LABEL: f4:
; CHECK: smlatt
; CHECK-THUMBV6-NOT: smlatt
        %tmp1 = ashr i32 %x, 16
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp3, %tmp1
        %tmp5 = add i32 %tmp4, %a
        ret i32 %tmp5
}

define i32 @f5(i32 %a, i16 %x, i16 %y) {
; CHECK-LABEL: f5:
; CHECK-NOT: sxth
; CHECK: smlabb
; CHECK-THUMBV6-NOT: smlabb
        %tmp1 = sext i16 %x to i32
        %tmp3 = sext i16 %y to i32
        %tmp4 = mul i32 %tmp3, %tmp1
        %tmp5 = add i32 %tmp4, %a
        ret i32 %tmp5
}

define i32 @f6(i32 %a, i32 %x, i16 %y) {
; CHECK-LABEL: f6:
; CHECK-NOT: sxth
; CHECK: {{smlatb r0, r1, r2, r0|smlabt r0, r2, r1, r0}}
; CHECK-THUMBV6-NOT: {{smlatb|smlabt}}
        %tmp1 = sext i16 %y to i32
        %tmp2 = ashr i32 %x, 16
        %tmp3 = mul i32 %tmp2, %tmp1
        %tmp5 = add i32 %tmp3, %a
        ret i32 %tmp5
}

define i32 @f7(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f7:
; CHECK: smlawb r0, r0, r1, r2
; CHECK-THUMBV6-NOT: smlawb
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
; CHECK-NOT: sxth
; CHECK: smlawb r0, r0, r1, r2
; CHECK-THUMBV6-NOT: smlawb
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
; CHECK: smlawt r0, r0, r1, r2
; CHECK-THUMBV6-NOT: smlawt
        %conv = sext i32 %a to i64
        %shr = ashr i32 %b, 16
        %conv1 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr26 = lshr i64 %mul, 16
        %conv3 = trunc i64 %shr26 to i32
        %add = add nsw i32 %conv3, %c
        ret i32 %add
}

define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: smulwb r0, r0, r1
; CHECK-THUMBV6-NOT: smulwb
        %shl = shl i32 %b, 16
        %shr = ashr exact i32 %shl, 16
        %conv = sext i32 %a to i64
        %conv2 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv2, %conv
        %shr37 = lshr i64 %mul, 16
        %conv4 = trunc i64 %shr37 to i32
        ret i32 %conv4
}

define i32 @f11(i32 %a, i16 signext %b) {
; CHECK-LABEL: f11:
; CHECK-NOT: sxth
; CHECK: smulwb r0, r0, r1
; CHECK-THUMBV6-NOT: smulwb
        %conv = sext i32 %a to i64
        %conv1 = sext i16 %b to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr4 = lshr i64 %mul, 16
        %conv2 = trunc i64 %shr4 to i32
        ret i32 %conv2
}

define i32 @f12(i32 %a, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: smulwt r0, r0, r1
; CHECK-THUMBV6-NOT: smulwt
        %conv = sext i32 %a to i64
        %shr = ashr i32 %b, 16
        %conv1 = sext i32 %shr to i64
        %mul = mul nsw i64 %conv1, %conv
        %shr25 = lshr i64 %mul, 16
        %conv3 = trunc i64 %shr25 to i32
        ret i32 %conv3
}

define i32 @f13(i32 %x, i16 %y) {
; CHECK-LABEL: f13:
; CHECK-NOT: sxth
; CHECK: {{smultb r0, r0, r1|smulbt r0, r1, r0}}
; CHECK-THUMBV6-NOT: {{smultb|smulbt}}
        %tmp1 = sext i16 %y to i32
        %tmp2 = ashr i32 %x, 16
        %tmp3 = mul i32 %tmp2, %tmp1
        ret i32 %tmp3
}

define i32 @f14(i32 %x, i32 %y) {
; CHECK-LABEL: f14:
; CHECK-NOT: sxth
; CHECK: {{smultb r0, r0, r1|smulbt r0, r1, r0}}
; CHECK-THUMBV6-NOT: {{smultb|smulbt}}
        %tmp1 = shl i32 %y, 16
        %tmp2 = ashr i32 %tmp1, 16
        %tmp3 = ashr i32 %x, 16
        %tmp4 = mul i32 %tmp3, %tmp2
        ret i32 %tmp4
}

define i32 @f15(i32 %x, i32 %y) {
; CHECK-LABEL: f15:
; CHECK-NOT: sxth
; CHECK: {{smulbt r0, r0, r1|smultb r0, r1, r0}}
; CHECK-THUMBV6-NOT: {{smulbt|smultb}}
        %tmp1 = shl i32 %x, 16
        %tmp2 = ashr i32 %tmp1, 16
        %tmp3 = ashr i32 %y, 16
        %tmp4 = mul i32 %tmp2, %tmp3
        ret i32 %tmp4
}

define i32 @f16(i16 %x, i16 %y) {
; CHECK-LABEL: f16:
; CHECK-NOT: sxth
; CHECK: smulbb
; CHECK-THUMBV6-NOT: smulbb
        %tmp1 = sext i16 %x to i32
        %tmp2 = sext i16 %x to i32
        %tmp3 = mul i32 %tmp1, %tmp2
        ret i32 %tmp3
}

define i32 @f17(i32 %x, i32 %y) {
; CHECK-LABEL: f17:
; CHECK: smulbb
; CHECK-THUMBV6-NOT: smulbb
        %tmp1 = shl i32 %x, 16
        %tmp2 = shl i32 %y, 16
        %tmp3 = ashr i32 %tmp1, 16
        %tmp4 = ashr i32 %tmp2, 16
        %tmp5 = mul i32 %tmp3, %tmp4
        ret i32 %tmp5
}

define i32 @f18(i32 %a, i32 %x, i32 %y) {
; CHECK-LABEL: f18:
; CHECK: {{smlabt r0, r1, r2, r0|smlatb r0, r2, r1, r0}}
; CHECK-THUMBV6-NOT: {{smlabt|smlatb}}
        %tmp0 = shl i32 %x, 16
        %tmp1 = ashr i32 %tmp0, 16
        %tmp2 = ashr i32 %y, 16
        %tmp3 = mul i32 %tmp2, %tmp1
        %tmp5 = add i32 %tmp3, %a
        ret i32 %tmp5
}

define i32 @f19(i32 %a, i32 %x, i32 %y) {
; CHECK-LABEL: f19:
; CHECK: {{smlatb r0, r1, r2, r0|smlabt r0, r2, r1, r0}}
; CHECK-THUMBV6-NOT: {{smlatb|smlabt}}
        %tmp0 = shl i32 %y, 16
        %tmp1 = ashr i32 %tmp0, 16
        %tmp2 = ashr i32 %x, 16
        %tmp3 = mul i32 %tmp2, %tmp1
        %tmp5 = add i32 %tmp3, %a
        ret i32 %tmp5
}

define i32 @f20(i32 %a, i32 %x, i32 %y) {
; CHECK-LABEL: f20:
; CHECK: smlabb
; CHECK-THUMBV6-NOT: smlabb
        %tmp1 = shl i32 %x, 16
        %tmp2 = ashr i32 %tmp1, 16
        %tmp3 = shl i32 %y, 16
        %tmp4 = ashr i32 %tmp3, 16
        %tmp5 = mul i32 %tmp2, %tmp4
        %tmp6 = add i32 %tmp5, %a
        ret i32 %tmp6
}

define i32 @f21(i32 %a, i32 %x, i16 %y) {
; CHECK-LABEL: f21
; CHECK-NOT: sxth
; CHECK: smlabb
; CHECK-THUMBV6-NOT: smlabb
        %tmp1 = shl i32 %x, 16
        %tmp2 = ashr i32 %tmp1, 16
        %tmp3 = sext i16 %y to i32
        %tmp4 = mul i32 %tmp2, %tmp3
        %tmp5 = add i32 %a, %tmp4
        ret i32 %tmp5
}

@global_b = external global i16, align 2

define i32 @f22(i32 %a) {
; CHECK-LABEL: f22:
; CHECK: smulwb r0, r0, r1
; CHECK-THUMBV6-NOT: smulwb
        %b = load i16, i16* @global_b, align 2
        %sext = sext i16 %b to i64
        %conv = sext i32 %a to i64
        %mul = mul nsw i64 %sext, %conv
        %shr37 = lshr i64 %mul, 16
        %conv4 = trunc i64 %shr37 to i32
        ret i32 %conv4
}

define i32 @f23(i32 %a, i32 %c) {
; CHECK-LABEL: f23:
; CHECK: smlawb r0, r0, r2, r1
; CHECK-THUMBV6-NOT: smlawb
        %b = load i16, i16* @global_b, align 2
        %sext = sext i16 %b to i64
        %conv = sext i32 %a to i64
        %mul = mul nsw i64 %sext, %conv
        %shr49 = lshr i64 %mul, 16
        %conv5 = trunc i64 %shr49 to i32
        %add = add nsw i32 %conv5, %c
        ret i32 %add
}
