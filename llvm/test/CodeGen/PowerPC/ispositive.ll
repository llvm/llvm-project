; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

; CHECK-LABEL: test1
; CHECK: srwi r3, r3, 31
; CHECK: blr
define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
}

