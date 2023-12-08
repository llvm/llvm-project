; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux \
; RUN: -mcpu=g5 < %s | FileCheck %s

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-freebsd \
; RUN: -mcpu=g5 < %s | FileCheck %s

; CHECK-LABEL: test_rol
; CHECK-NOT:     spr
; CHECK-NOT:     vrsave
; CHECK:         vrlw
; CHECK-NEXT:    blr
define <4 x i32> @test_rol() {
        ret <4 x i32> < i32 -11534337, i32 -11534337, i32 -11534337, i32 -11534337 >
}

; CHECK-LABEL: test_arg
; CHECK-NOT:     spr
; CHECK-NOT:     vrsave
define <4 x i32> @test_arg(<4 x i32> %A, <4 x i32> %B) {
        %C = add <4 x i32> %A, %B               ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %C
}

