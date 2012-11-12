; RUN: llc < %s -march=thumb | FileCheck %s -check-prefix=V5
; RUN: llc < %s -march=thumb -mattr=+v6 | FileCheck %s -check-prefix=V6

; rdar://7176514

define i32 @test1(i8* %t1) nounwind {
; V5: ldrb

; V6: ldrb
    %tmp.u = load i8* %t1
    %tmp1.s = zext i8 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test2(i16* %t1) nounwind {
; V5: ldrh

; V6: ldrh
    %tmp.u = load i16* %t1
    %tmp1.s = zext i16 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test3(i8* %t0) nounwind {
; V5: ldrb
; V5: lsls
; V5: asrs

; V6: ldrb
; V6: sxtb
    %tmp.s = load i8* %t0
    %tmp1.s = sext i8 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test4(i16* %t0) nounwind {
; V5: ldrh
; V5: lsls
; V5: asrs

; V6: ldrh
; V6: sxth
    %tmp.s = load i16* %t0
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test5() nounwind {
; V5: movs r0, #0
; V5: ldrsh

; V6: movs r0, #0
; V6: ldrsh
    %tmp.s = load i16* null
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}
