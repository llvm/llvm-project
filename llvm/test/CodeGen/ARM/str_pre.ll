; RUN: llc -mtriple=arm-eabi %s -o -  | FileCheck %s

define void @test1(ptr %X, ptr %A, ptr %dest) {
        %B = load i32, ptr %A               ; <i32> [#uses=1]
        %Y = getelementptr i32, ptr %X, i32 4               ; <ptr> [#uses=2]
        store i32 %B, ptr %Y
        store ptr %Y, ptr %dest
        ret void
}

define ptr @test2(ptr %X, ptr %A) {
        %B = load i32, ptr %A               ; <i32> [#uses=1]
        %Y = getelementptr i16, ptr %X, i32 4               ; <ptr> [#uses=2]
        %tmp = trunc i32 %B to i16              ; <i16> [#uses=1]
        store i16 %tmp, ptr %Y
        ret ptr %Y
}

; CHECK: str{{.*}}!
; CHECK: str{{.*}}!
; CHECK-NOT: str{{.*}}!

