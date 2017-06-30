; RUN: llc < %s -march=arm -mtriple=arm-linux-gnu | FileCheck %s

define void @f(i32 %a) {
entry:
; CHECK: add  r11, sp, #8
        %tmp = alloca i8, i32 %a                ; <i8*> [#uses=1]
        call void @g( i8* %tmp, i32 %a, i32 1, i32 2, i32 3 )
        ret void
; CHECK: sub  sp, r11, #8
}

declare void @g(i8*, i32, i32, i32, i32)
