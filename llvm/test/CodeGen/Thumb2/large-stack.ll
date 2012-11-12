; RUN: llc < %s -march=thumb -mattr=+thumb2 -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -march=thumb -mattr=+thumb2 -mtriple=arm-linux-gnueabi | FileCheck %s -check-prefix=LINUX

define void @test1() {
; DARWIN: test1:
; DARWIN: sub sp, #256
; LINUX: test1:
; LINUX: sub sp, #256
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; DARWIN: test2:
; DARWIN: sub.w sp, sp, #4160
; DARWIN: sub sp, #8
; LINUX: test2:
; LINUX: sub.w sp, sp, #4160
; LINUX: sub sp, #8
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
; DARWIN: test3:
; DARWIN: push    {r4, r7, lr}
; DARWIN: sub.w sp, sp, #805306368
; DARWIN: sub sp, #20
; LINUX: test3:
; LINUX: push.w {r4, r7, r11, lr}
; LINUX: sub.w sp, sp, #805306368
; LINUX: sub sp, #16
    %retval = alloca i32, align 4
    %tmp = alloca i32, align 4
    %a = alloca [805306369 x i8], align 16
    store i32 0, i32* %tmp
    %tmp1 = load i32* %tmp
    ret i32 %tmp1
}
