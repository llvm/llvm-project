; RUN: llc < %s -mtriple=armv7-apple-ios -o - | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7m-darwin-eabi -o - | FileCheck %s --check-prefix=DARWIN
; RUN: llc < %s -mtriple=arm-none-eabi -o - | FileCheck --check-prefix=EABI %s

@from = common global [500 x i32] zeroinitializer, align 4
@to = common global [500 x i32] zeroinitializer, align 4

define void @f() {
entry:

        ; CHECK: memmove
        ; EABI: __aeabi_memmove
        call void @llvm.memmove.p0i8.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0, i1 false)

        ; CHECK: memcpy
        ; EABI: __aeabi_memcpy
        call void @llvm.memcpy.p0i8.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8* bitcast ([500 x i32]* @to to i8*), i32 500, i32 0, i1 false)

        ; EABI memset swaps arguments
        ; CHECK: mov r1, #0
        ; CHECK: memset
        ; DARWIN: movs r1, #0
        ; DARWIN: memset
        ; EABI: mov r2, #0
        ; EABI: __aeabi_memset
        call void @llvm.memset.p0i8.i32(i8* bitcast ([500 x i32]* @from to i8*), i8 0, i32 500, i32 0, i1 false)
        unreachable
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
