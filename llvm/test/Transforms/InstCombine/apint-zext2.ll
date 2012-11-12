; Tests to make sure elimination of casts is working correctly
; This test is for Integer BitWidth > 64 && BitWidth <= 1024.
; RUN: opt < %s -instcombine -S | FileCheck %s

define i1024 @test_sext_zext(i77 %A) {
    %c1 = zext i77 %A to i533
    %c2 = sext i533 %c1 to i1024
    ret i1024 %c2
; CHECK: %c2 = zext i77 %A to i1024
; CHECK: ret i1024 %c2
}
