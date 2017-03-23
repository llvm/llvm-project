; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s

define i32 @test1(i32 %X) {
entry:
; CHECK-LABEL: test1:
; CHECK: lsrs r0, r0, #31
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

define i32 @test2(i32 %X) {
entry:
; CHECK-LABEL: test2:
; CHECK: lsls r1, r1, #31
; CHECK-NEXT: adds
        %tmp1 = sub i32 %X, 2147483648
        ret i32 %tmp1
}

