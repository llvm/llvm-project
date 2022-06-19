; RUN: opt < %s -passes=hwasan -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use(i8*, i8*)

define void @test_alloca() sanitize_hwaddress {
  ; CHECK: alloca { [4 x i8], [12 x i8] }, align 16
  %x = alloca i8, i64 4
  ; CHECK: alloca i8, i64 16, align 16
  %y = alloca i8, i64 16
  call void @use(i8* %x, i8* %y)
  ret void
}
