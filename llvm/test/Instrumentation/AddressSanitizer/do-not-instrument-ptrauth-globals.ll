; This test checks that we are not instrumenting llvm.ptrauth globals.
; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

declare void @f()

@f.ptrauth.ia.42 = private constant { ptr, i32, i64, i64 } { ptr @f, i32 0, i64 0, i64 42 }, section "llvm.ptrauth"

; CHECK: @f.ptrauth.ia.42 = private constant { ptr, i32, i64, i64 } { ptr @f, i32 0, i64 0, i64 42 }, section "llvm.ptrauth"
