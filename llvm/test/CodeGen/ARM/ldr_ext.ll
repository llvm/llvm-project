; RUN: llc < %s -march=arm | FileCheck %s

define i32 @test1(i8* %t1) nounwind {
; CHECK: ldrb
    %tmp.u = load i8* %t1
    %tmp1.s = zext i8 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test2(i16* %t1) nounwind {
; CHECK: ldrh
    %tmp.u = load i16* %t1
    %tmp1.s = zext i16 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test3(i8* %t0) nounwind {
; CHECK: ldrsb
    %tmp.s = load i8* %t0
    %tmp1.s = sext i8 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test4(i16* %t0) nounwind {
; CHECK: ldrsh
    %tmp.s = load i16* %t0
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test5() nounwind {
; CHECK: mov r0, #0
; CHECK: ldrsh
    %tmp.s = load i16* null
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}
