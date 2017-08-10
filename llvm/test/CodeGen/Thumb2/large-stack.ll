; RUN: llc < %s -mcpu=arm1156t2-s -mattr=+thumb2 -mtriple=thumb-apple-darwin \
; RUN:    | FileCheck %s -check-prefix=DARWIN
; RUN: llc < %s -mcpu=arm1156t2-s -mattr=+thumb2 -mtriple=thumb-linux-gnueabi \
; RUN:    | FileCheck %s -check-prefix=LINUX

define void @test1() {
; DARWIN-LABEL: test1:
; DARWIN: sub sp, #256
; LINUX-LABEL: test1:
; LINUX: sub sp, #256
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; DARWIN-LABEL: test2:
; DARWIN: sub.w sp, sp, #4160
; DARWIN: sub sp, #8
; LINUX-LABEL: test2:
; LINUX: sub.w sp, sp, #4160
; LINUX: sub sp, #8
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
; DARWIN-LABEL: test3:
; DARWIN: push    {r4, r7, lr}
; DARWIN: sub.w sp, sp, #805306368
; DARWIN: sub sp, #20
; LINUX-LABEL: test3:
; LINUX: push {r4, r6, r7, lr}
; LINUX: sub.w sp, sp, #805306368
; LINUX: sub sp, #16
    %retval = alloca i32, align 4
    %tmp = alloca i32, align 4
    %a = alloca [805306369 x i8], align 16
    store i32 0, i32* %tmp
    %tmp1 = load i32, i32* %tmp
    ret i32 %tmp1
}
