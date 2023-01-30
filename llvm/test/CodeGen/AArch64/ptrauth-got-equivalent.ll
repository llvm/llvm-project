; RUN: llc %s -o - -mtriple=arm64e-apple-ios | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

@foo = external global i64, align 8
@foo.ptrauth = private constant { ptr, i32, i64, i64 } { ptr @foo, i32 2, i64 0, i64 12345 }, section "llvm.ptrauth", align 8

; CHECK-LABEL: l_got.foo:
; CHECK-NEXT: 	.quad	_foo@AUTH(da,12345)

; CHECK-LABEL: _foo_ref:
; CHECK-NEXT: 	.long	l_got.foo-_foo_ref

@got.foo = private unnamed_addr constant ptr @foo.ptrauth
@foo_ref = constant i32 trunc (i64 sub (i64 ptrtoint (ptr @got.foo to i64), i64 ptrtoint (ptr @foo_ref to i64)) to i32), align 8
