; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a, i32* %v) {
; CHECK: f1:
; CHECK: str r0, [r1]
        store i32 %a, i32* %v
        ret i32 %a
}

define i32 @f2(i32 %a, i32* %v) {
; CHECK: f2:
; CHECK: str.w r0, [r1, #4092]
        %tmp2 = getelementptr i32* %v, i32 1023
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f2a(i32 %a, i32* %v) {
; CHECK: f2a:
; CHECK: str r0, [r1, #-128]
        %tmp2 = getelementptr i32* %v, i32 -32
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f3(i32 %a, i32* %v) {
; CHECK: f3:
; CHECK: mov.w r2, #4096
; CHECK: str r0, [r1, r2]
        %tmp2 = getelementptr i32* %v, i32 1024
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f4(i32 %a, i32 %base) {
entry:
; CHECK: f4:
; CHECK: str r0, [r1, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i32*
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f5(i32 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f5:
; CHECK: str r0, [r1, r2]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i32*
        store i32 %a, i32* %tmp2
        ret i32 %a
}

define i32 @f6(i32 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f6:
; CHECK: str.w r0, [r1, r2, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        store i32 %a, i32* %tmp3
        ret i32 %a
}

define i32 @f7(i32 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f7:
; CHECK: lsrs r2, r2, #2
; CHECK: str r0, [r1, r2]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        store i32 %a, i32* %tmp3
        ret i32 %a
}
