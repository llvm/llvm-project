; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i8 @f1(i8 %a, i8* %v) {
; CHECK: f1:
; CHECK: strb r0, [r1]
        store i8 %a, i8* %v
        ret i8 %a
}

define i8 @f2(i8 %a, i8* %v) {
; CHECK: f2:
; CHECK: strb.w r0, [r1, #4092]
        %tmp2 = getelementptr i8* %v, i32 4092
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f2a(i8 %a, i8* %v) {
; CHECK: f2a:
; CHECK: strb r0, [r1, #-128]
        %tmp2 = getelementptr i8* %v, i32 -128
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f3(i8 %a, i8* %v) {
; CHECK: f3:
; CHECK: mov.w r2, #4096
; CHECK: strb r0, [r1, r2]
        %tmp2 = getelementptr i8* %v, i32 4096
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f4(i8 %a, i32 %base) {
entry:
; CHECK: f4:
; CHECK: strb r0, [r1, #-128]
        %tmp1 = sub i32 %base, 128
        %tmp2 = inttoptr i32 %tmp1 to i8*
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f5(i8 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f5:
; CHECK: strb r0, [r1, r2]
        %tmp1 = add i32 %base, %offset
        %tmp2 = inttoptr i32 %tmp1 to i8*
        store i8 %a, i8* %tmp2
        ret i8 %a
}

define i8 @f6(i8 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f6:
; CHECK: strb.w r0, [r1, r2, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        store i8 %a, i8* %tmp3
        ret i8 %a
}

define i8 @f7(i8 %a, i32 %base, i32 %offset) {
entry:
; CHECK: f7:
; CHECK: lsrs r2, r2, #2
; CHECK: strb r0, [r1, r2]
        %tmp1 = lshr i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i8*
        store i8 %a, i8* %tmp3
        ret i8 %a
}
