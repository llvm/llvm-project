; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s -check-prefix=V5
; RUN: llc -mtriple=thumb-eabi -mattr=+v6 %s -o - | FileCheck %s -check-prefix=V6

; rdar://7176514

define i32 @test1(ptr %t1) nounwind {
; V5: ldrb

; V6: ldrb
    %tmp.u = load i8, ptr %t1
    %tmp1.s = zext i8 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test2(ptr %t1) nounwind {
; V5: ldrh

; V6: ldrh
    %tmp.u = load i16, ptr %t1
    %tmp1.s = zext i16 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test3(ptr %t0) nounwind {
; V5: ldrb
; V5: lsls
; V5: asrs

; V6: mov
; V6: ldrsb
    %tmp.s = load i8, ptr %t0
    %tmp1.s = sext i8 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test4(ptr %t0) nounwind {
; V5: ldrh
; V5: lsls
; V5: asrs

; V6: mov
; V6: ldrsh
    %tmp.s = load i16, ptr %t0
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test5() nounwind {
; V5: movs r0, #0
; V5: ldrsh

; V6: movs r0, #0
; V6: ldrsh
    %tmp.s = load i16, ptr null
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}
