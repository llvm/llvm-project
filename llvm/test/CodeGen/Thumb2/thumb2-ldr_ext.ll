; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @test1(ptr %v.pntr.s0.u1) {
    %tmp.u = load i8, ptr %v.pntr.s0.u1
    %tmp1.s = zext i8 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test2(ptr %v.pntr.s0.u1) {
    %tmp.u = load i16, ptr %v.pntr.s0.u1
    %tmp1.s = zext i16 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test3(ptr %v.pntr.s1.u0) {
    %tmp.s = load i8, ptr %v.pntr.s1.u0
    %tmp1.s = sext i8 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test4() {
    %tmp.s = load i16, ptr null
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}

; CHECK: ldrb
; CHECK-NOT: ldrb

; CHECK: ldrh
; CHECK-NOT: ldrh

; CHECK: ldrsb
; CHECK-NOT: ldrsb

; CHECK: ldrsh
; CHECK-NOT: ldrsh

