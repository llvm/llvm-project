; Tests to make sure elimination of casts is working correctly
; This test is for Integer BitWidth <= 64 && BitWidth % 2 != 0.
; RUN: opt < %s -instcombine -S | FileCheck %s

define i47 @test_sext_zext(i11 %A) {
    %c1 = zext i11 %A to i39
    %c2 = sext i39 %c1 to i47
    ret i47 %c2
; CHECK: %c2 = zext i11 %A to i47
; CHECK: ret i47 %c2
}
