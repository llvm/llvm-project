; RUN: llc < %s -mtriple=thumb-apple-ios | FileCheck %s

define void @test1() {
; CHECK: test1:
; CHECK: sub sp, #256
; CHECK: add sp, #256
    %tmp = alloca [ 64 x i32 ] , align 4
    ret void
}

define void @test2() {
; CHECK: test2:
; CHECK: ldr.n r0, LCPI
; CHECK: add sp, r0
; CHECK: subs r4, r7, #4
; CHECK: mov sp, r4
    %tmp = alloca [ 4168 x i8 ] , align 4
    ret void
}

define i32 @test3() {
; CHECK: test3:
; CHECK: ldr.n r2, LCPI
; CHECK: add sp, r2
; CHECK: ldr.n r1, LCPI
; CHECK: add r1, sp
; CHECK: subs r4, r7, #4
; CHECK: mov sp, r4
    %retval = alloca i32, align 4
    %tmp = alloca i32, align 4
    %a = alloca [805306369 x i8], align 16
    store i32 0, i32* %tmp
    %tmp1 = load i32* %tmp
    ret i32 %tmp1
}
