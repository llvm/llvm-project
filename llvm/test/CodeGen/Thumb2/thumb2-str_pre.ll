; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define void @test1(ptr %X, ptr %A, ptr %dest) {
; CHECK: test1
; CHECK: str  r1, [r0, #16]!
        %B = load i32, ptr %A               ; <i32> [#uses=1]
        %Y = getelementptr i32, ptr %X, i32 4               ; <ptr> [#uses=2]
        store i32 %B, ptr %Y
        store ptr %Y, ptr %dest
        ret void
}

define ptr @test2(ptr %X, ptr %A) {
; CHECK: test2
; CHECK: strh r1, [r0, #8]!
        %B = load i32, ptr %A               ; <i32> [#uses=1]
        %Y = getelementptr i16, ptr %X, i32 4               ; <ptr> [#uses=2]
        %tmp = trunc i32 %B to i16              ; <i16> [#uses=1]
        store i16 %tmp, ptr %Y
        ret ptr %Y
}
