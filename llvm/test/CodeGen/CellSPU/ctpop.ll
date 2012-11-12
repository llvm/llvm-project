; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep cntb    %t1.s | count 3
; RUN: grep andi    %t1.s | count 3
; RUN: grep rotmi   %t1.s | count 2
; RUN: grep rothmi  %t1.s | count 1
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

declare i8 @llvm.ctpop.i8(i8)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctpop.i32(i32)

define i32 @test_i8(i8 %X) {
        call i8 @llvm.ctpop.i8(i8 %X)
        %Y = zext i8 %1 to i32
        ret i32 %Y
}

define i32 @test_i16(i16 %X) {
        call i16 @llvm.ctpop.i16(i16 %X)
        %Y = zext i16 %1 to i32
        ret i32 %Y
}

define i32 @test_i32(i32 %X) {
        call i32 @llvm.ctpop.i32(i32 %X)
        %Y = bitcast i32 %1 to i32
        ret i32 %Y
}

