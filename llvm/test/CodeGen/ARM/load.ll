; RUN: llc < %s -march=arm > %t
; RUN: grep ldrsb %t
; RUN: grep ldrb %t
; RUN: grep ldrsh %t
; RUN: grep ldrh %t


define i32 @f1(i8* %p) {
entry:
        %tmp = load i8* %p              ; <i8> [#uses=1]
        %tmp1 = sext i8 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @f2(i8* %p) {
entry:
        %tmp = load i8* %p              ; <i8> [#uses=1]
        %tmp2 = zext i8 %tmp to i32              ; <i32> [#uses=1]
        ret i32 %tmp2
}

define i32 @f3(i16* %p) {
entry:
        %tmp = load i16* %p             ; <i16> [#uses=1]
        %tmp3 = sext i16 %tmp to i32             ; <i32> [#uses=1]
        ret i32 %tmp3
}

define i32 @f4(i16* %p) {
entry:
        %tmp = load i16* %p             ; <i16> [#uses=1]
        %tmp4 = zext i16 %tmp to i32             ; <i32> [#uses=1]
        ret i32 %tmp4
}
