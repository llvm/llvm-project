; RUN: llc < %s -mtriple=arm-linux-gnu | FileCheck %s

define void @f(i32 %a) {
entry:
; CHECK: add  r11, sp, #8
        %tmp = alloca i8, i32 %a                ; <ptr> [#uses=1]
        call void @g( ptr %tmp, i32 %a, i32 1, i32 2, i32 3 )
        ret void
; CHECK: sub  sp, r11, #8
}

declare void @g(ptr, i32, i32, i32, i32)
