; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i16 @f1(i16* %v) {
entry:
; CHECK: f1:
; CHECK: ldrh r0, [r0]
        %tmp = load i16* %v
        ret i16 %tmp
}

define i16 @f2(i16* %v) {
entry:
; CHECK: f2:
; CHECK: ldrh.w r0, [r0, #2046]
        %tmp2 = getelementptr i16* %v, i16 1023
        %tmp = load i16* %tmp2
        ret i16 %tmp
}

define i16 @f3(i16* %v) {
entry:
; CHECK: f3:
; CHECK: mov.w r1, #4096
; CHECK: ldrh r0, [r0, r1]
        %tmp2 = getelementptr i16* %v, i16 2048
        %tmp = load i16* %tmp2
        ret i16 %tmp
}

define i16 @f4(i32 %base) {
entry:
; CHECK: f4:
; CHECK: ldrh r0, [r0, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i16*
        %tmp3 = load i16* %tmp2
        ret i16 %tmp3
}

define i16 @f5(i32 %base, i32 %offset) {
entry:
; CHECK: f5:
; CHECK: ldrh r0, [r0, r1]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i16*
        %tmp3 = load i16* %tmp2
        ret i16 %tmp3
}

define i16 @f6(i32 %base, i32 %offset) {
entry:
; CHECK: f6:
; CHECK: ldrh.w r0, [r0, r1, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        %tmp4 = load i16* %tmp3
        ret i16 %tmp4
}

define i16 @f7(i32 %base, i32 %offset) {
entry:
; CHECK: f7:
; CHECK: lsrs r1, r1, #2
; CHECK: ldrh r0, [r0, r1]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i16*
        %tmp4 = load i16* %tmp3
        ret i16 %tmp4
}
