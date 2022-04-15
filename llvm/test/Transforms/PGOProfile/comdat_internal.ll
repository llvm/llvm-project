; RUN: opt < %s -passes=pgo-instr-gen,instrprof -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$foo = comdat any
; CHECK: $foo = comdat any

; CHECK: $__llvm_profile_raw_version = comdat any
; CHECK: $__profc__stdin__foo.[[#FOO_HASH:]] = comdat any

@bar = global i32 ()* @foo, align 8

; CHECK: @__llvm_profile_raw_version = constant i64 {{[0-9]+}}, comdat
; CHECK-NOT: __profn__stdin__foo
; CHECK: @__profc__stdin__foo.[[#FOO_HASH]] = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; CHECK: @__profd__stdin__foo.[[#FOO_HASH]] = private global { i64, i64, i64, i8*, i8*, i32, [2 x i16] } { i64 -5640069336071256030, i64 [[#FOO_HASH]], i64 sub (i64 ptrtoint ([1 x i64]* @__profc__stdin__foo.742261418966908927 to i64), i64 ptrtoint ({ i64, i64, i64, i8*, i8*, i32, [2 x i16] }* @__profd__stdin__foo.742261418966908927 to i64)), i8* null
; CHECK-NOT: bitcast (i32 ()* @foo to i8*)
; CHECK-SAME: , i8* null, i32 1, [2 x i16] zeroinitializer }, section "__llvm_prf_data", comdat($__profc__stdin__foo.[[#FOO_HASH]]), align 8
; CHECK: @__llvm_prf_nm
; CHECK: @llvm.compiler.used

define internal i32 @foo() comdat {
entry:
  ret i32 1
}
