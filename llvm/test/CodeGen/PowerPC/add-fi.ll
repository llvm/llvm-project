; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define ptr @test1() {
        %X = alloca { i32, i32 }
        %Y = getelementptr {i32,i32}, ptr %X, i32 0, i32 1
        ret ptr %Y

; CHECK-LABEL: @test1
; CHECK: addi 3, 1, -4
; CHECK: blr
}

define ptr @test2() {
        %X = alloca { i32, i32, i32, i32 }
        %Y = getelementptr {i32,i32,i32,i32}, ptr %X, i32 0, i32 3
        ret ptr %Y

; CHECK-LABEL: @test2
; CHECK: addi 3, 1, -4
; CHECK: blr
}

