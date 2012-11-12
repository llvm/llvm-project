; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32* %v) {
entry:
; CHECK: f1:
; CHECK: ldr r0, [r0]
        %tmp = load i32* %v
        ret i32 %tmp
}

define i32 @f2(i32* %v) {
entry:
; CHECK: f2:
; CHECK: ldr.w r0, [r0, #4092]
        %tmp2 = getelementptr i32* %v, i32 1023
        %tmp = load i32* %tmp2
        ret i32 %tmp
}

define i32 @f3(i32* %v) {
entry:
; CHECK: f3:
; CHECK: mov.w r1, #4096
; CHECK: ldr r0, [r0, r1]
        %tmp2 = getelementptr i32* %v, i32 1024
        %tmp = load i32* %tmp2
        ret i32 %tmp
}

define i32 @f4(i32 %base) {
entry:
; CHECK: f4:
; CHECK: ldr r0, [r0, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i32*
        %tmp3 = load i32* %tmp2
        ret i32 %tmp3
}

define i32 @f5(i32 %base, i32 %offset) {
entry:
; CHECK: f5:
; CHECK: ldr r0, [r0, r1]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i32*
        %tmp3 = load i32* %tmp2
        ret i32 %tmp3
}

define i32 @f6(i32 %base, i32 %offset) {
entry:
; CHECK: f6:
; CHECK: ldr.w r0, [r0, r1, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = load i32* %tmp3
        ret i32 %tmp4
}

define i32 @f7(i32 %base, i32 %offset) {
entry:
; CHECK: f7:
; CHECK: lsrs r1, r1, #2
; CHECK: ldr r0, [r0, r1]

        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = load i32* %tmp3
        ret i32 %tmp4
}
