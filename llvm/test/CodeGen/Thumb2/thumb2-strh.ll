; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i16 @f1(i16 %a, ptr %v) {
; CHECK-LABEL: f1:
; CHECK: strh r0, [r1]
        store i16 %a, ptr %v
        ret i16 %a
}

define i16 @f2(i16 %a, ptr %v) {
; CHECK-LABEL: f2:
; CHECK: strh.w r0, [r1, #4092]
        %tmp2 = getelementptr i16, ptr %v, i32 2046
        store i16 %a, ptr %tmp2
        ret i16 %a
}

define i16 @f2a(i16 %a, ptr %v) {
; CHECK-LABEL: f2a:
; CHECK: strh r0, [r1, #-128]
        %tmp2 = getelementptr i16, ptr %v, i32 -64
        store i16 %a, ptr %tmp2
        ret i16 %a
}

define i16 @f3(i16 %a, ptr %v) {
; CHECK-LABEL: f3:
; CHECK: mov.w r2, #4096
; CHECK: strh r0, [r1, r2]
        %tmp2 = getelementptr i16, ptr %v, i32 2048
        store i16 %a, ptr %tmp2
        ret i16 %a
}

define i16 @f4(i16 %a, i32 %base) {
entry:
; CHECK-LABEL: f4:
; CHECK: strh r0, [r1, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to ptr
        store i16 %a, ptr %tmp2
        ret i16 %a
}

define i16 @f5(i16 %a, i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f5:
; CHECK: strh r0, [r1, r2]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to ptr
        store i16 %a, ptr %tmp2
        ret i16 %a
}

define i16 @f6(i16 %a, i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f6:
; CHECK: strh.w r0, [r1, r2, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        store i16 %a, ptr %tmp3
        ret i16 %a
}

define i16 @f7(i16 %a, i32 %base, i32 %offset) {
entry:
; CHECK-LABEL: f7:
; CHECK: lsrs r2, r2, #2
; CHECK: strh r0, [r1, r2]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to ptr
        store i16 %a, ptr %tmp3
        ret i16 %a
}
