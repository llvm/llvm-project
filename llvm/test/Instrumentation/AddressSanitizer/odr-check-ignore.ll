; RUN: opt < %s -asan -asan-module -S | FileCheck %s 

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global [2 x i32] zeroinitializer, align 4
@b = private global [2 x i32] zeroinitializer, align 4
@c = internal global [2 x i32] zeroinitializer, align 4
@d = unnamed_addr global [2 x i32] zeroinitializer, align 4

; CHECK: @__asan_global_a = private global { i64, i64, i64, i64, i64, i64, i64, i64 } { i64 ptrtoint ({ [2 x i32], [56 x i8] }* @a to i64), i64 8, i64 64, i64 ptrtoint ([2 x i8]* @___asan_gen_.1 to i64), i64 ptrtoint ([8 x i8]* @___asan_gen_ to i64), i64 0, i64 0, i64 0 }

; CHECK: @__asan_global_b = private global { i64, i64, i64, i64, i64, i64, i64, i64 } { i64 ptrtoint ({ [2 x i32], [56 x i8] }* @b to i64), i64 8, i64 64, i64 ptrtoint ([2 x i8]* @___asan_gen_.2 to i64), i64 ptrtoint ([8 x i8]* @___asan_gen_ to i64), i64 0, i64 0, i64 -1 }

; CHECK: @__asan_global_c = private global { i64, i64, i64, i64, i64, i64, i64, i64 } { i64 ptrtoint ({ [2 x i32], [56 x i8] }* @c to i64), i64 8, i64 64, i64 ptrtoint ([2 x i8]* @___asan_gen_.3 to i64), i64 ptrtoint ([8 x i8]* @___asan_gen_ to i64), i64 0, i64 0, i64 -1 }

; CHECK: @__asan_global_d = private global { i64, i64, i64, i64, i64, i64, i64, i64 } { i64 ptrtoint ({ [2 x i32], [56 x i8] }* @d to i64), i64 8, i64 64, i64 ptrtoint ([2 x i8]* @___asan_gen_.4 to i64), i64 ptrtoint ([8 x i8]* @___asan_gen_ to i64), i64 0, i64 0, i64 0 }
