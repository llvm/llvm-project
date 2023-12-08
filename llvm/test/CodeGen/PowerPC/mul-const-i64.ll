; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=generic < %s -mtriple=ppc64-- | FileCheck %s -check-prefix=GENERIC-CHECK
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr8 < %s -mtriple=ppc64-- | FileCheck %s -check-prefixes=PWR8-CHECK,CHECK
; RUN: llc -verify-machineinstrs -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr -mcpu=pwr9 < %s -mtriple=ppc64le-- | FileCheck %s -check-prefixes=PWR9-CHECK,CHECK


define i64 @foo(i64 %a) {
entry:
  %mul = mul nsw i64 %a, 6
  ret i64 %mul
}

; GENERIC-CHECK-LABEL: @foo
; GENERIC-CHECK: mulli r3, r3, 6
; GENERIC-CHECK: blr

define i64 @test1(i64 %a) {
        %tmp.1 = mul nsw i64 %a, 16         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test1:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 4


define i64 @test2(i64 %a) {
        %tmp.1 = mul nsw i64 %a, 17         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test2:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: add r[[REG2:[0-9]+]], r3, r[[REG1]]

define i64 @test3(i64 %a) {
        %tmp.1 = mul nsw i64 %a, 15         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test3:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r[[REG1]], r3

; negtive constant

define i64 @test4(i64 %a) {
        %tmp.1 = mul nsw i64 %a, -16         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test4:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: neg r[[REG2:[0-9]+]], r[[REG1]]

define i64 @test5(i64 %a) {
        %tmp.1 = mul nsw i64 %a, -17         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test5:
; PWR9-CHECK: mulli r[[REG1:[0-9]+]], r3, -17
; PWR8-CHECK-NOT: mul
; PWR8-CHECK: sldi r[[REG1:[0-9]+]], r3, 4
; PWR8-CHECK-NEXT: add r[[REG2:[0-9]+]], r3, r[[REG1]]
; PWR8-CHECK-NEXT: neg r{{[0-9]+}}, r[[REG2]]

define i64 @test6(i64 %a) {
        %tmp.1 = mul nsw i64 %a, -15         ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test6:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 4
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r3, r[[REG1]]
; CHECK-NOT: neg

; boundary case

define i64 @test7(i64 %a) {
        %tmp.1 = mul nsw i64 %a, -9223372036854775808 ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test7:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 63

define i64 @test8(i64 %a) {
        %tmp.1 = mul nsw i64 %a, 9223372036854775807 ; <i64> [#uses=1]
        ret i64 %tmp.1
}
; CHECK-LABEL: test8:
; CHECK-NOT: mul
; CHECK: sldi r[[REG1:[0-9]+]], r3, 63
; CHECK-NEXT: sub r[[REG2:[0-9]+]], r[[REG1]], r3
