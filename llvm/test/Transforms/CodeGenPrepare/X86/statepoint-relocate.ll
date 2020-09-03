; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()

define i32 @test_sor_basic(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr)]
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_two_derived(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base, i32 12
; CHECK: getelementptr i32, i32* %base-new, i32 12
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 12
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr, i32* %ptr2)]
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 2)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_ooo(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr)]
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_smallint([3 x i32]* %base) gc "statepoint-example" {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
; CHECK: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 2
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"([3 x i32]* %base, i32* %ptr)]
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(token %tok, i32 0, i32 0)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_largeint([3 x i32]* %base) gc "statepoint-example" {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
; CHECK-NOT: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 21
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"([3 x i32]* %base, i32* %ptr)]
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(token %tok, i32 0, i32 0)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_noop(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 2)
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 12
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr, i32* %ptr2)]
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 2)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_basic_wrong_order(i32* %base) gc "statepoint-example" {
; CHECK-LABEL: @test_sor_basic_wrong_order
; Here we have base relocate inserted after derived. Make sure that we don't
; produce uses of the relocated base pointer before it's definition.
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       ; CHECK: getelementptr i32, i32* %base, i32 15
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr)]
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       ; CHECK: %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       ; CHECK-NEXT: getelementptr i32, i32* %base-new, i32 15
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_noop_cross_bb(i1 %external-cond, i32* %base) gc "statepoint-example" {
; CHECK-LABEL: @test_sor_noop_cross_bb
; Here base relocate doesn't dominate derived relocate. Make sure that we don't
; produce undefined use of the relocated base pointer.
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       ; CHECK: getelementptr i32, i32* %base, i32 15
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr)]
       br i1 %external-cond, label %left, label %right

left:
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       ; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ret-new = load i32, i32* %ptr-new
       ret i32 %ret-new

right:
       %ptr-base = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       ; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ret-base = load i32, i32* %ptr-base
       ret i32 %ret-base
}

define i32 @test_sor_noop_same_bb(i1 %external-cond, i32* %base) gc "statepoint-example" {
; CHECK-LABEL: @test_sor_noop_same_bb
; Here base relocate doesn't dominate derived relocate. Make sure that we don't
; produce undefined use of the relocated base pointer.
entry:
       %ptr1 = getelementptr i32, i32* %base, i32 15
       ; CHECK: getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 5
       ; CHECK: getelementptr i32, i32* %base, i32 5
       %tok = call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live"(i32* %base, i32* %ptr1, i32* %ptr2)]
       ; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 2)
       %ret2-new = load i32, i32* %ptr2-new
       ; CHECK: getelementptr i32, i32* %base-new, i32 5
       %ptr1-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 1)
       %ret1-new = load i32, i32* %ptr1-new
       ; CHECK: getelementptr i32, i32* %base-new, i32 15
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(token %tok, i32 0, i32 0)
       %ret-new = add i32 %ret2-new, %ret1-new
       ret i32 %ret-new
}

declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32* @llvm.experimental.gc.relocate.p0i32(token, i32, i32)
declare [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(token, i32, i32)
