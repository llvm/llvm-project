; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i8 @f1(ptr %v) {
entry:
; CHECK-LABEL: f1:
; CHECK: ldrb r0, [r0]
        %tmp = load i8, ptr %v
        ret i8 %tmp
}

define i8 @f2(ptr %v) {
entry:
; CHECK-LABEL: f2:
; CHECK: ldrb r0, [r0, #-1]
        %tmp2 = getelementptr i8, ptr %v, i8 1023
        %tmp = load i8, ptr %tmp2
        ret i8 %tmp
}

define i8 @f3(i32 %base) {
entry:
; CHECK-LABEL: f3:
; CHECK: mov.w r1, #4096
; CHECK: ldrb r0, [r0, r1]
        %tmp1 = add i32 %base, 4096
        %tmp2 = inttoptr i32 %tmp1 to ptr
        %tmp3 = load i8, ptr %tmp2
        ret i8 %tmp3
}

define i8 @f4(i32 %base) {
entry:
; CHECK-LABEL: f4:
; CHECK: ldrb r0, [r0, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to ptr
        %tmp3 = load i8, ptr %tmp2
        ret i8 %tmp3
}

define i8 @f5(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f5:
; CHECK: ldrb r0, [r0, r1]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to ptr
        %tmp3 = load i8, ptr %tmp2
        ret i8 %tmp3
}

define i8 @f6(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f6:
; CHECK: ldrb.w r0, [r0, r1, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        %tmp4 = load i8, ptr %tmp3
        ret i8 %tmp4
}

define i8 @f7(i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f7:
; CHECK: lsrs r1, r1, #2
; CHECK: ldrb r0, [r0, r1]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        %tmp4 = load i8, ptr %tmp3
        ret i8 %tmp4
}
