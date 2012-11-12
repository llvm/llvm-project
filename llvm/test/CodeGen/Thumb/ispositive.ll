; RUN: llc < %s -march=thumb | FileCheck %s

define i32 @test1(i32 %X) {
entry:
; CHECK: test1:
; CHECK: lsrs r0, r0, #31
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

