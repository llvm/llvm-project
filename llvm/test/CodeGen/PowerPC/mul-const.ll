; RUN: llc -verify-machineinstrs -mcpu=pwr8 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s -mtriple=ppc64-- | FileCheck %s -check-prefixes=PWR8-CHECK,CHECK
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s -mtriple=ppc64le-- | FileCheck %s -check-prefixes=PWR9-CHECK,CHECK

define i32 @test1(i32 %a) {
        %tmp.1 = mul nsw i32 %a, 16         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test1:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 4

define i32 @test2(i32 %a) {
        %tmp.1 = mul nsw i32 %a, 17         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test2:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: add r[[REG2:[0-9]+]], r3, r[[REG1]]

define i32 @test3(i32 %a) {
        %tmp.1 = mul nsw i32 %a, 15         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test3:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r[[REG1]], r3

; negtive constant

define i32 @test4(i32 %a) {
        %tmp.1 = mul nsw i32 %a, -16         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test4:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: neg r[[REG2:[0-9]+]], r[[REG1]]

define i32 @test5(i32 %a) {
        %tmp.1 = mul nsw i32 %a, -17         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test5:
; PWR9-CHECK: mulli r[[REG1:[0-9]+]], r3, -17
; PWR8-CHECK-NOT: mul
; PWR8-CHECK: slwi r[[REG1:[0-9]+]], r3, 4
; PWR8-CHECK-NEXT: add r[[REG2:[0-9]+]], r3, r[[REG1]]
; PWR8-CHECK-NEXT: neg r{{[0-9]+}}, r[[REG2]]

define i32 @test6(i32 %a) {
        %tmp.1 = mul nsw i32 %a, -15         ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test6:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r3, r[[REG1]]
; CHECK-NOT: neg

; boundary case

define i32 @test7(i32 %a) {
        %tmp.1 = mul nsw i32 %a, -2147483648 ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test7:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 31

define i32 @test8(i32 %a) {
        %tmp.1 = mul nsw i32 %a, 2147483647 ; <i32> [#uses=1]
        ret i32 %tmp.1
}
; CHECK-LABEL: test8:
; CHECK-NOT: mul
; CHECK: slwi r[[REG1:[0-9]+]], r3, 31
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r[[REG1]], r3
