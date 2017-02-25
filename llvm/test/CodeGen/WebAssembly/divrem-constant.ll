; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that integer div and rem by constant are optimized appropriately.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; CHECK-LABEL: test_udiv_2:
; CHECK: i32.shr_u
define i32 @test_udiv_2(i32 %x) {
    %t = udiv i32 %x, 2
    ret i32 %t
}

; CHECK-LABEL: test_udiv_5:
; CHECK: i32.div_u
define i32 @test_udiv_5(i32 %x) {
    %t = udiv i32 %x, 5
    ret i32 %t
}

; CHECK-LABEL: test_sdiv_2:
; CHECK: i32.div_s
define i32 @test_sdiv_2(i32 %x) {
    %t = sdiv i32 %x, 2
    ret i32 %t
}

; CHECK-LABEL: test_sdiv_5:
; CHECK: i32.div_s
define i32 @test_sdiv_5(i32 %x) {
    %t = sdiv i32 %x, 5
    ret i32 %t
}

; CHECK-LABEL: test_urem_2:
; CHECK: i32.and
define i32 @test_urem_2(i32 %x) {
    %t = urem i32 %x, 2
    ret i32 %t
}

; CHECK-LABEL: test_urem_5:
; CHECK: i32.rem_u
define i32 @test_urem_5(i32 %x) {
    %t = urem i32 %x, 5
    ret i32 %t
}

; CHECK-LABEL: test_srem_2:
; CHECK: i32.rem_s
define i32 @test_srem_2(i32 %x) {
    %t = srem i32 %x, 2
    ret i32 %t
}

; CHECK-LABEL: test_srem_5:
; CHECK: i32.rem_s
define i32 @test_srem_5(i32 %x) {
    %t = srem i32 %x, 5
    ret i32 %t
}
