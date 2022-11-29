; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(ptr, ptr)

define void @test_alloca() sanitize_hwaddress {
  ; CHECK: alloca { [4 x i8], [12 x i8] }, align 16
  %x = alloca i8, i64 4
  ; CHECK: alloca i8, i64 16, align 16
  %y = alloca i8, i64 16
  call void @use(ptr %x, ptr %y)
  ret void
}
