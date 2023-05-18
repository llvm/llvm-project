; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i16 @f1(ptr %v) {
entry:
; CHECK-LABEL: f1:
; CHECK: ldrh r0, [r0]
        %tmp = load i16, ptr %v
        ret i16 %tmp
}

define i16 @f2(ptr %v) {
entry:
; CHECK-LABEL: f2:
; CHECK: ldrh.w r0, [r0, #2046]
        %tmp2 = getelementptr i16, ptr %v, i16 1023
        %tmp = load i16, ptr %tmp2
        ret i16 %tmp
}

define i16 @f3(ptr %v) {
entry:
; CHECK-LABEL: f3:
; CHECK: mov.w r1, #4096
; CHECK: ldrh r0, [r0, r1]
        %tmp2 = getelementptr i16, ptr %v, i16 2048
        %tmp = load i16, ptr %tmp2
        ret i16 %tmp
}

define i16 @f4(i32 %base) {
entry:
; CHECK-LABEL: f4:
; CHECK: ldrh r0, [r0, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to ptr
        %tmp3 = load i16, ptr %tmp2
        ret i16 %tmp3
}

define i16 @f5(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f5:
; CHECK: ldrh r0, [r0, r1]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to ptr
        %tmp3 = load i16, ptr %tmp2
        ret i16 %tmp3
}

define i16 @f6(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f6:
; CHECK: ldrh.w r0, [r0, r1, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        %tmp4 = load i16, ptr %tmp3
        ret i16 %tmp4
}

define i16 @f7(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f7:
; CHECK: lsrs r1, r1, #2
; CHECK: ldrh r0, [r0, r1]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        %tmp4 = load i16, ptr %tmp3
        ret i16 %tmp4
}
