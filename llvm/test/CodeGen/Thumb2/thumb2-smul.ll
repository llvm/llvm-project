; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+dsp %s -o - |  FileCheck %s

@x = weak global i16 0          ; <ptr> [#uses=1]
@y = weak global i16 0          ; <ptr> [#uses=0]

define i32 @f1(i32 %y) {
; CHECK: f1
; CHECK: smulbt r0, r1, r0
        %tmp = load i16, ptr @x             ; <i16> [#uses=1]
        %tmp1 = add i16 %tmp, 2         ; <i16> [#uses=1]
        %tmp2 = sext i16 %tmp1 to i32           ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp2, %tmp3            ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @f2(i32 %x, i32 %y) {
; CHECK: f2
; CHECK: smultt r0, r1, r0
        %tmp1 = ashr i32 %x, 16         ; <i32> [#uses=1]
        %tmp3 = ashr i32 %y, 16         ; <i32> [#uses=1]
        %tmp4 = mul i32 %tmp3, %tmp1            ; <i32> [#uses=1]
        ret i32 %tmp4
}
