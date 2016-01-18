; RUN: opt -codegenprepare -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()

define i32 @test_sor_basic(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok,  i32 7, i32 7)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
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
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr, i32* %ptr2)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok,  i32 7, i32 7)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 9)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_ooo(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: getelementptr i32, i32* %base-new, i32 15
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok,  i32 7, i32 7)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_smallint([3 x i32]* %base) gc "statepoint-example" {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
; CHECK: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 2
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 2
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, [3 x i32]* %base, i32* %ptr)
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32 %tok,  i32 7, i32 7)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_gep_largeint([3 x i32]* %base) gc "statepoint-example" {
; CHECK: getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
; CHECK-NOT: getelementptr [3 x i32], [3 x i32]* %base-new, i32 0, i32 21
entry:
       %ptr = getelementptr [3 x i32], [3 x i32]* %base, i32 0, i32 21
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, [3 x i32]* %base, i32* %ptr)
       %base-new = call [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32 %tok,  i32 7, i32 7)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %ret = load i32, i32* %ptr-new
       ret i32 %ret
}

define i32 @test_sor_noop(i32* %base) gc "statepoint-example" {
; CHECK: getelementptr i32, i32* %base, i32 15
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 9)
entry:
       %ptr = getelementptr i32, i32* %base, i32 15
       %ptr2 = getelementptr i32, i32* %base, i32 12
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr, i32* %ptr2)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %ptr2-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 9)
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
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok,  i32 7, i32 7)
       ; CHECK: %base-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok,  i32 7, i32 7)
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
       %tok = call i32 (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0, i32* %base, i32* %ptr)
       br i1 %external-cond, label %left, label %right

left:
       %ptr-new = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       ; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 8)
       %ret-new = load i32, i32* %ptr-new
       ret i32 %ret-new

right:
       %ptr-base = call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 7)
       ; CHECK: call i32* @llvm.experimental.gc.relocate.p0i32(i32 %tok, i32 7, i32 7)
       %ret-base = load i32, i32* %ptr-base
       ret i32 %ret-base
}

declare i32 @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32* @llvm.experimental.gc.relocate.p0i32(i32, i32, i32)
declare [3 x i32]* @llvm.experimental.gc.relocate.p0a3i32(i32, i32, i32)
