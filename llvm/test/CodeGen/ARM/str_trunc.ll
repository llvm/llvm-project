; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define void @test1(i32 %v, ptr %ptr) {
        %tmp = trunc i32 %v to i16              ; <i16> [#uses=1]
        store i16 %tmp, ptr %ptr
        ret void
}

define void @test2(i32 %v, ptr %ptr) {
        %tmp = trunc i32 %v to i8               ; <i8> [#uses=1]
        store i8 %tmp, ptr %ptr
        ret void
}

; CHECK: strh
; CHECK-NOT: strh

; CHECK: strb
; CHECK-NOT: strb

