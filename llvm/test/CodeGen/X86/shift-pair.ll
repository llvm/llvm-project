; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

define i64 @test(i64 %A) {
; CHECK: @test
; CHECK: shrq $54
; CHECK: andl $1020
; CHECK: ret
    %B = lshr i64 %A, 56
    %C = shl i64 %B, 2
    ret i64 %C
}
